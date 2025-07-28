import argparse
from datasets import load_dataset
from reward import reward_func, format_reward_func, validity_reward_func, retrieval_reward_func_v3
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import socket
import glob
import wandb
import torch


def train_grpo(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    batch_size: int,
    learning_rate: float,
    use_vllm: bool,
    load_in_8bit: bool,
    vllm_mode: str,
    train_lora: bool,
    max_completion_length: int,
    max_prompt_length: int,
    temperature: float,
    gradient_checkpointing: bool,
    num_iterations: int,
    num_generations: int,
    epochs: int,
    logging_steps: int,
    gradient_accumulation_steps: int,
    seed: int,
    sample: bool=False,
    alpha: float=1.0,
):
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(lambda x: {"alpha": alpha})
    if sample:
        # Add a column with the length of ground_truth
        def compute_gt_length(example):
            return {"gt_length": len(example["ground_truth"])}

        dataset = dataset.map(compute_gt_length)

        # Sort by the new 'gt_length' column in descending order
        dataset = dataset.sort("gt_length", reverse=True)

        # Select the top 30,000 examples
        dataset = dataset.select(range(min(30000, len(dataset))))

        # Shuffle the selected subset
        dataset = dataset.shuffle(seed=seed)


    print(f"Using dataset: {dataset_name}")
    print(f"Using model: {model_name}")
    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"  # spreads across GPUs if available
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    if train_lora:
        lora_config = LoraConfig(
            base_model_name_or_path=model_name,
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        model = get_peft_model(base_model, lora_config)
        # calculate the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")
    else:
        model = base_model
    print("Model loaded successfully.")
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        bf16=True,
        save_steps=100,
        num_iterations=num_iterations,
        gradient_checkpointing=gradient_checkpointing,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        per_device_train_batch_size=batch_size,
        temperature=temperature,
        beta=0.001,
        num_generations=num_generations,
        log_completions=True,
        num_train_epochs= epochs,
        deepspeed="ds_config.json",
        report_to="wandb",
        vllm_gpu_memory_utilization=0.5,
        scale_rewards=False,
        mask_truncated_completions=True,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_vllm=use_vllm,
        ds3_gather_for_generation=False,
        vllm_mode=vllm_mode,
        use_liger_loss=True
    )

    print(f"config: {config}")
    try:
        eval_dataset = load_dataset(dataset_name, split="test")
        eval_dataset = eval_dataset.map(lambda x: {"alpha": alpha})
        # sample maximum 5 examples from the eval dataset
        eval_dataset = eval_dataset.shuffle(seed=seed).select(range(min(5, len(eval_dataset))))
        trainer = GRPOTrainer(
            model=model,
            args=config,
            reward_funcs=[format_reward_func, validity_reward_func, retrieval_reward_func_v3],
            # tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
        )
    except ValueError:
        trainer = GRPOTrainer(
            model=model,
            args=config,
            reward_funcs=[format_reward_func, validity_reward_func, retrieval_reward_func_v3],
            train_dataset=dataset
        )

    # import wandb
    #
    #
    # wandb.init(
    #     project="huggingface",
    #     name=f"grpo-run-{model_name}-{dataset_name}",
    #     mode="online"  # or "offline" if training without internet access
    # )

    print("Trainer initialized successfully.")
    valid_checkpoint = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if len(valid_checkpoint) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # save the model to final
    output_final_dir = os.path.join(output_dir, "final")
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(output_final_dir):
            os.makedirs(output_final_dir)
        trainer.model.save_pretrained(output_final_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRPO with a Boolean query reward function.")

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="wshuai190/pubmed-pmc-oa-sr-dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo-boolean-query")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate")
    parser.add_argument("--train_lora", action="store_true", help="Whether to train LoRA layers")
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=12)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    #add load in 8 bit
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load the model in 8-bit mode")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="Used by deepspeed for distributed training")
    # add sample bool
    parser.add_argument("--sample", action="store_true")

    args = parser.parse_args()

    print(f"Training GRPO with the following parameters: {args}")
    train_grpo(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        epochs=args.epochs,
        temperature=args.temperature,
        load_in_8bit=args.load_in_8bit,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        num_iterations=args.num_iterations,
        gradient_checkpointing=args.gradient_checkpointing,
        num_generations=args.num_generations,
        logging_steps=args.logging_steps,
        train_lora = args.train_lora,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        sample=args.sample,
        alpha=args.alpha
    )



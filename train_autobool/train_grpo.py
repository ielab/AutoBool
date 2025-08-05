"""
GRPO Training Script for Boolean Query Generation

This script trains a language model using Group Relative Policy Optimization (GRPO)
to generate high-quality Boolean queries for medical literature search.
"""

import argparse
import os
import glob
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from reward import format_reward_func, validity_reward_func, retrieval_reward_func
from utils.logging_config import setup_training_logger


class ModelConfig:
    """Configuration for model loading and training setup."""

    def __init__(
            self,
            model_name: str,
            load_in_8bit: bool = False,
            train_lora: bool = False,
            gradient_checkpointing: bool = False
    ):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.train_lora = train_lora
        self.gradient_checkpointing = gradient_checkpointing

    def create_model(self):
        """Create and configure the model based on settings."""
        if self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )

        if self.train_lora:
            lora_config = LoraConfig(
                base_model_name_or_path=self.model_name,
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ],
            )
            model = get_peft_model(base_model, lora_config)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger = setup_training_logger("model_setup", "logs")
            logger.info(f"Number of trainable parameters: {trainable_params:,}")
        else:
            model = base_model

        return model


def prepare_dataset(dataset_name: str, alpha: float, sample: bool, seed: int, max_samples: int = 30000):
    """Load and prepare the training dataset."""
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(lambda x: {"alpha": alpha})

    if sample:
        # Sort by ground truth length and take top samples
        dataset = dataset.map(lambda x: {"gt_length": len(x["ground_truth"])})
        dataset = dataset.sort("gt_length", reverse=True)
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        dataset = dataset.shuffle(seed=seed)

    return dataset


def prepare_eval_dataset(dataset_name: str, alpha: float, seed: int, max_eval_samples: int = 5):
    """Load and prepare the evaluation dataset."""
    try:
        eval_dataset = load_dataset(dataset_name, split="test")
        eval_dataset = eval_dataset.map(lambda x: {"alpha": alpha})
        eval_dataset = eval_dataset.shuffle(seed=seed).select(
            range(min(max_eval_samples, len(eval_dataset)))
        )
        return eval_dataset
    except ValueError:
        logger = setup_training_logger("dataset_prep", "logs")
        logger.warning("No test split found, training without evaluation dataset")
        return None


def create_grpo_config(
        output_dir: str,
        learning_rate: float,
        batch_size: int,
        max_completion_length: int,
        max_prompt_length: int,
        temperature: float,
        num_iterations: int,
        num_generations: int,
        epochs: int,
        logging_steps: int,
        gradient_accumulation_steps: int,
        gradient_checkpointing: bool,
        use_vllm: bool,
        vllm_mode: str
) -> GRPOConfig:
    """Create GRPO training configuration."""
    return GRPOConfig(
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
        num_train_epochs=epochs,
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


def train_grpo(
        model_name: str,
        dataset_name: str,
        output_dir: str,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        use_vllm: bool = False,
        load_in_8bit: bool = False,
        vllm_mode: str = "colocate",
        train_lora: bool = False,
        max_completion_length: int = 2048,
        max_prompt_length: int = 1536,
        temperature: float = 0.6,
        gradient_checkpointing: bool = False,
        num_iterations: int = 1,
        num_generations: int = 12,
        epochs: int = 1,
        logging_steps: int = 50,
        gradient_accumulation_steps: int = 2,
        seed: int = 42,
        sample: bool = False,
        alpha: float = 1.0,
):
    """Main training function."""
    # Setup training logger
    logger = setup_training_logger("grpo_training", output_dir)
    
    logger.info("Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Alpha: {alpha}")
    logger.info(f"  Sample Dataset: {sample}")

    # Prepare datasets
    train_dataset = prepare_dataset(dataset_name, alpha, sample, seed)
    eval_dataset = prepare_eval_dataset(dataset_name, alpha, seed)
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation samples: {len(eval_dataset)}")

    # Create model
    model_config = ModelConfig(model_name, load_in_8bit, train_lora, gradient_checkpointing)
    model = model_config.create_model()
    logger.info("Model loaded successfully.")

    # Create training configuration
    config = create_grpo_config(
        output_dir, learning_rate, batch_size, max_completion_length,
        max_prompt_length, temperature, num_iterations, num_generations,
        epochs, logging_steps, gradient_accumulation_steps,
        gradient_checkpointing, use_vllm, vllm_mode
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[format_reward_func, validity_reward_func, retrieval_reward_func],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    logger.info("Trainer initialized successfully.")

    # Check for existing checkpoints
    existing_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    resume_from_checkpoint = len(existing_checkpoints) > 0

    if resume_from_checkpoint:
        logger.info("Resuming training from existing checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info("Starting training from scratch...")
        trainer.train()

    # Save final model
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        output_final_dir = os.path.join(output_dir, "final")
        os.makedirs(output_final_dir, exist_ok=True)
        trainer.model.save_pretrained(output_final_dir)
        logger.info(f"Final model saved to: {output_final_dir}")
    elif not torch.distributed.is_initialized():
        output_final_dir = os.path.join(output_dir, "final")
        os.makedirs(output_final_dir, exist_ok=True)
        trainer.model.save_pretrained(output_final_dir)
        logger.info(f"Final model saved to: {output_final_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GRPO with Boolean query reward functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data arguments
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--dataset_name", type=str,
                        default="wshuai190/pubmed-pmc-oa-sr-dataset",
                        help="HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/grpo-boolean-query",
                        help="Output directory for checkpoints")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per device training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of GRPO iterations")
    parser.add_argument("--num_generations", type=int, default=12,
                        help="Number of generations per prompt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging frequency")

    # Model configuration
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit mode")
    parser.add_argument("--train_lora", action="store_true",
                        help="Train with LoRA adapters")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")

    # Generation arguments
    parser.add_argument("--max_completion_length", type=int, default=2048,
                        help="Maximum completion length")
    parser.add_argument("--max_prompt_length", type=int, default=1536,
                        help="Maximum prompt length")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")

    # VLLM arguments
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use VLLM for generation")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["colocate", "separate"],
                        help="VLLM execution mode")

    # Dataset and reward arguments
    parser.add_argument("--sample", action="store_true",
                        help="Sample subset of training data")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha parameter for recall weighting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    logger = setup_training_logger("main", "logs")
    logger.info("=" * 60)
    logger.info("GRPO Boolean Query Training")
    logger.info("=" * 60)

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
        train_lora=args.train_lora,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        sample=args.sample,
        alpha=args.alpha
    )
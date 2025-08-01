# AutoBool

**AutoBool** is a reinforcement learning (RL) framework for training large language models (LLMs) to generate high-quality Boolean queries for systematic reviews. It supports an end-to-end pipeline—from data collection and preprocessing to query generation and evaluation—tailored to high-recall retrieval tasks in biomedical literature search (e.g., PubMed).

---

## 📁 Project Structure

### `data/`
- Raw and intermediate data storage.

### `dataset_prepare/`
Scripts to prepare training/evaluation/temporal1000 datasets.

- `process_pubmed.py`: Processes PubMed data into a structured format for data createion (train, test, temporal1000).
- `upload_to_hf.py`: Uploads processed datasets to Hugging Face for easy access.
- `process_clef.py `: Processes CLEF tar data for boolean query generation evaluation.
- `process_seed,py`: Processes seed collection data for boolean query generation evaluation.
'
---

### `train_autobool/`
Scripts for training and evaluating the AutoBool model.

- `train_grpo.py`: Main training script using GRPO (Group Relative Policy Optimization).
- `reward.py`: Reward function combining syntactic validity, format correctness, and retrieval effectiveness.
- `run_generation.py`: Boolean query generation and execution pipeline.
- `run_generation_chatgpt.py`: Baseline generation using OpenAI models.
- `compute_additional_metrics.py`: Computes evaluation metrics such as Recall, F₃, Precision, Success Rate, etc.
- Dataset creation by prompt type:
  - `create_dataset_no_reason.py`
  - `create_dataset_reasoning.py`
  - `create_dataset_conceptual.py`
  - `create_dataset_objective.py`
- `utils.py`: Shared training utilities.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets  
*(Skip this step if datasets are already prepared)*

```bash
python dataset_prepare/process_pubmed.py --steps 0,1,2,3,4,5
python dataset_prepare/upload_to_hf.py ../data/processed/pubmed/sr_augmented/all.jsonl your_hf_dataset_name 
```

Note huggingface datasets are already available uploaded, you can skip the dataset preparation step and directly use the following command to upload the dataset:

```bash
huggingface-cli download [anonymoused_for_review] 
```


### 3. Train AutoBool Model

```bash
deepspeed --include localhost:${gpu_list} --master_port $master_port \
  train_grpo_v3.py \
  --max_completion_length $max_completion_length \
  --max_prompt_length $max_prompt_length \
  --train_lora \
  --use_vllm \
  --alpha $alpha \
  --num_generations 4 \
  --temperature $temperature \
  --batch_size $batch_size \
  --logging_steps 50 \
  --gradient_checkpointing \
  --epochs 1 \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --dataset_name $dataset \
  --model_name $model_name \
  --output_dir $output_dir 
    
```
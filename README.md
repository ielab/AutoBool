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
- `entrez_api/`: FastAPI service for PubMed query processing (see dedicated README)

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

### 3. Start Entrez API Service
*Required for training - the reward function needs access to PubMed queries*

```bash
cd train_autobool/entrez_api

# Configure API keys in entrez_submission_api.py first
# Then start the service
docker-compose up --build -d
```

For detailed API setup instructions, see `train_autobool/entrez_api/README.md`.

### 4. Create Training Dataset
*Format your processed data for training with different prompt types*

```bash
cd train_autobool

# Create dataset with reasoning prompts (includes <think></think> tags)
python create_unified_dataset.py --prompt-type reasoning \
  --data-path ../data/processed/pubmed/sr_augmented_result \
  --hf-name your-username/pubmed-reasoning-dataset

# Create dataset with no reasoning (direct answer only)
python create_unified_dataset.py --prompt-type no_reason \
  --data-path ../data/processed/pubmed/sr_augmented_result \
  --hf-name your-username/pubmed-no-reason-dataset

# Create dataset with conceptual method
python create_unified_dataset.py --prompt-type conceptual \
  --data-path ../data/processed/pubmed/sr_augmented_result \
  --hf-name your-username/pubmed-conceptual-dataset

# For CLEF/SEED datasets (no splits)
python create_unified_dataset.py --prompt-type reasoning \
  --data-path ../data/processed/clef_augmented \
  --hf-name your-username/clef-reasoning-dataset \
  --no-split
```

**Available prompt types:**
- `no_reason`: Direct Boolean query generation without reasoning
- `reasoning`: Includes reasoning process in `<think></think>` tags
- `conceptual`: Step-by-step conceptual method approach
- `objective`: Objective method with simulated examples

### 5. Train AutoBool Model

```bash
deepspeed --include localhost:${gpu_list} --master_port $master_port \
  train_grpo.py \
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
  --dataset_name your-username/pubmed-reasoning-dataset \
  --model_name $model_name \
  --output_dir $output_dir 
    
```

**Note:** Use the dataset name from step 4 in the `--dataset_name` parameter.
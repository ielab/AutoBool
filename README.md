# AutoBool

**AutoBool** is a reinforcement learning (RL) framework for training large language models (LLMs) to generate high-quality Boolean queries for systematic reviews. It supports end-to-end pipeline—from data collection and preprocessing to query generation and evaluation—tailored to high-recall retrieval tasks in biomedical literature search (e.g., PubMed).

---

## 📁 Project Structure

### `data/`
- Raw and intermediate data storage.

### `dataset_prepare/`
Scripts to prepare training and evaluation datasets.

- `01_download_pubmed.py`: Fetches source data from PubMed (PMC Open Access).
- `02_split_review_all.py`: Splits systematic reviews into training and test subsets.
- `03_process_clef.py`, `03_process_seed.py`: Prepares CLEF TAR and Seed Collection datasets.
- `03_split_by_review_types.py`: Categorizes reviews based on methodology type.
- `04_date_correction_retrieve.py`: Applies date-based filtering (e.g., Temporal-Cutoff).
- `05_sr_collection_remove_clef_seed_ids.py`: Removes overlap with CLEF/Seed topics.
- `06_filter_references.py`: Cleans up reference and inclusion lists.
- `utils/`: Helper utilities (e.g., `00_bool_query_ids.py` for mapping queries to PMIDs).

---

### `train_autobool/`
Scripts for training and evaluating the AutoBool model.

- `train_grpo_v3.py`: Main training script using GRPO (Group Relative Policy Optimization).
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

### 2. Prepare Datasets (You can skip this step as datasets are already provided)
```bash
python process_pubmed.py
python3 process_clef.py
python3 process_seed.py
```



### 3. Train AutoBool Model

```bash
python train_autobool/train_grpo_v3.py 
```


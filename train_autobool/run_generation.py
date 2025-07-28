#!/usr/bin/env python3
"""Fully-fixed evaluation script that works with or without vLLM.
Changes from the original version
──────────────────────────────────
1. Multiprocessing start-method is only set when **not** using vLLM.
2. Dataset.map() replaced with an in-process batching loop so the model / vLLM
   engine are never pickled.
3. Duplicate, wrong `final_dataset` block removed – recall/precision columns
   are present.
4. retrieve_batch keeps output order when using ThreadPoolExecutor.
5. Added generation-success log (`generation_success_log.tsv`).
6. Minor quality-of-life tweaks: chat-template fallback, clearer logging.
"""

from __future__ import annotations
from tqdm import tqdm
import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from reward import check_logic
import torch
from datasets import load_dataset

import multiprocessing as mp

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CUDA determinism (optional but helps)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _maybe_set_spawn(use_vllm: bool) -> None:
    if not use_vllm and mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

def compute_recall_precision(
    retrieved_ids: List[str], reference_pmids: List[str]
) -> Tuple[float, float]:
    if not retrieved_ids or not reference_pmids:
        print(0, 0, len(retrieved_ids))
        return 0.0, 0.0
    retrieved_set, reference_set = set(retrieved_ids), set(reference_pmids)
    tp = retrieved_set & reference_set
    recall = len(tp) / len(reference_set) if reference_set else 0.0
    precision = len(tp) / len(retrieved_set) if retrieved_set else 0.0
    print(recall, precision, len(retrieved_set))
    return recall, precision

def extract_boolean_query(output: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    boolean_query = match.group(1).strip() if match else ""
    return boolean_query


def _render_prompt(tokenizer, prompt: List | str) -> str:
    if isinstance(prompt, List):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)  # type: ignore[arg-type]
    else:
        return prompt if isinstance(prompt, str) else json.dumps(prompt)

def generate_batch(
    model,
    tokenizer,
    prompts: List,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    use_vllm: bool = False,
    vllm_engine=None,
) -> List[str]:
    rendered_prompts = [_render_prompt(tokenizer, p) for p in prompts]

    if use_vllm:
        if vllm_engine is None or SamplingParams is None:
            raise RuntimeError("vLLM engine or SamplingParams is not initialised.")

        sampling = SamplingParams(
            max_tokens=max_completion_length, temperature=temperature, top_p=top_p, seed=42
        )
        results = vllm_engine.generate(rendered_prompts, sampling, use_tqdm=False)
        generated = []
        for prompt_text, gen in zip(rendered_prompts, results):
            full = gen.outputs[0].text  # type: ignore[index]
            generated.append(full[len(prompt_text):].strip() if full.startswith(prompt_text) else full.strip())
        return generated

    enc = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)

    input_lens = enc["input_ids"].shape[1]
    with torch.no_grad():
        out_tokens = model.generate(
            **enc,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for seq in out_tokens:
        gen = seq[input_lens:]
        results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return results

def process_batch(
    batch: Dict[str, List],
    model,
    tokenizer,
    args,
    vllm_engine=None,
) -> Dict[str, List]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from reward import cached_retrieve_documents
    except ImportError as exc:
        raise RuntimeError("The `reward` module is missing: {}".format(exc)) from exc

    outputs = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=batch["prompt"],
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=args.use_vllm,
        vllm_engine=vllm_engine,
    )
    queries = [extract_boolean_query(o) for o in outputs]
    print(queries)
    if args.ref_date:
        # Use the reference date for retrieval if specified
        mindates = batch.get("first_ref_date", ["1945/01/01"] * len(queries))
        maxdates = batch.get("last_ref_date", ["2025/01/01"] * len(queries))
    else:
        mindates = batch.get("min-date", ["1945/01/01"] * len(queries))
        maxdates = batch.get("max-date", ["2025/01/01"] * len(queries))
    retrieved_ids: List[List[str]] = [None] * len(queries)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=min(len(queries), 2)) as pool:
        future_map = {
            pool.submit(cached_retrieve_documents, q, mi, ma): idx
            for idx, (q, mi, ma) in enumerate(zip(queries, mindates, maxdates))
        }
        for fut in as_completed(future_map):
            i = future_map[fut]
            try:
                ids = fut.result()
            except Exception as exc:
                print(f"[WARN] Retrieval failed for query index {i}: {exc}")
                ids = []
            retrieved_ids[i] = ids

    recalls, precisions = zip(
        *[compute_recall_precision(ret, gt) for ret, gt in zip(retrieved_ids, batch["ground_truth"])]
    )

    return {
        "topicid": batch["topicid"],
        "output": outputs,
        "boolean_query": queries,
        "retrieved_ids": retrieved_ids,
        "recall": list(recalls),
        "precision": list(precisions),
    }

def merge_peft_model_and_save(model_path: str) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig

    print(f"Merging PEFT adapter found at: {model_path}")
    try:
        cfg = PeftConfig.from_pretrained(model_path)
        base_name = cfg.base_model_name_or_path
        print(f"Detected base model: {base_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to("cuda")

        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)

        backup_dir = os.path.join(model_path, "lora_backup")
        os.makedirs(backup_dir, exist_ok=True)

        for fname in os.listdir(model_path):
            fpath = os.path.join(model_path, fname)
            dest = os.path.join(backup_dir, fname)
            if os.path.isfile(fpath) or os.path.islink(fpath):
                shutil.move(fpath, dest)
            elif os.path.isdir(fpath) and fname != "lora_backup":
                shutil.move(fpath, dest)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        del model, base_model
        torch.cuda.empty_cache()
        print("[✓] Merge complete – merged model saved. Original LoRA files backed up in 'lora_backup/'.")

    except Exception:
        print("[!] Failed to merge PEFT model – is this already a full model?")

def run_evaluation_loop(dataset, model, tokenizer, args, vllm_engine):
    max_retries = 10
    attempts = defaultdict(int)
    success_flags = defaultdict(bool)
    topicid_to_example = {row["topicid"]: dict(row) for row in dataset}
    pending_ids = list(topicid_to_example.keys())
    generation_log: Dict[int, int] = {}

    for round_no in range(1, max_retries + 1):
        if not pending_ids:
            print("All topics processed – exiting loop.")
            break
        print(f"\n=== Round {round_no}/{max_retries} – {len(pending_ids)} topics ===")

        examples = [topicid_to_example[tid] for tid in pending_ids]
        batch_results: Dict[str, List] = defaultdict(list)

        for idx in tqdm(range(0, len(examples), args.batch_size)):
            mini = examples[idx : idx + args.batch_size]
            mini_batch = {k: [ex[k] for ex in mini] for k in examples[0]}
            res = process_batch(mini_batch, model, tokenizer, args, vllm_engine)
            for key, val in res.items():
                batch_results[key].extend(val)

        newly_successful: List[str] = []
        for i, topicid in enumerate(batch_results["topicid"]):
            retrieved = batch_results["retrieved_ids"][i]
            if retrieved and (not success_flags[topicid]):
                success_flags[topicid] = True
                newly_successful.append(topicid)
                topicid_to_example[topicid].update(
                    {
                        "output": batch_results["output"][i],
                        "boolean_query": batch_results["boolean_query"][i],
                        "retrieved_ids": retrieved,
                        "recall": batch_results["recall"][i],
                        "precision": batch_results["precision"][i],
                    }
                )
            elif round_no == max_retries:
                # If this is the last round, we still want to keep the output
                topicid_to_example[topicid].update(
                    {
                        "output": batch_results["output"][i],
                        "boolean_query": batch_results["boolean_query"][i],
                        "retrieved_ids": retrieved,
                        "recall": batch_results["recall"][i],
                        "precision": batch_results["precision"][i],
                    }
                )

        pending_ids = [tid for tid in pending_ids if not success_flags[tid]]
        print(f"[✓] {len(newly_successful)} topics solved this round | {len(pending_ids)} pending")
        generation_log[round_no] = len(newly_successful)
        if not newly_successful and pending_ids and round_no < max_retries:
            print("No progress this round – sleeping 2 s before retry …")
            time.sleep(2)

    final_dataset_list = [topicid_to_example[tid] for tid, ok in success_flags.items() if ok]

    # for all other topics, need to add empty fields


    if not final_dataset_list:
        print("[!] No topics succeeded – aborting metric calculation.")
        return None, generation_log

    from datasets import Dataset
    return Dataset.from_list(final_dataset_list), generation_log

def main(args):
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.sample:
        dataset = dataset.shuffle(seed=42).select(range(min(args.sample, len(dataset))))
    print(f"Loaded {len(dataset)} rows from {args.dataset_name}:{args.split}")

    _maybe_set_spawn(args.use_vllm)

    model = tokenizer = vllm_engine = None

    if args.use_vllm:
        if LLM is None:
            raise RuntimeError("--use_vllm but the vllm package isn't installed.")
        try:
            print(f"[INFO] Loading model with vLLM: {args.model_path}")
            vllm_engine = LLM(model=args.model_path, dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        except Exception as exc:
            print(f"[WARN] vLLM failed ({exc}) – falling back to HF model.")
            args.use_vllm = False

    if not args.use_vllm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token




    final_dataset, generation_log = run_evaluation_loop(dataset, model, tokenizer, args, vllm_engine)
    if final_dataset is None:
        out_dir = os.path.join("runs", os.path.relpath(os.path.normpath(args.model_path), os.getcwd()))
        os.makedirs(out_dir, exist_ok=True)
        gen_path = os.path.join(out_dir, "generation_success_log.tsv")
        with open(gen_path, "w", encoding="utf-8") as fp:
            fp.write("topicid\tnum_generations_to_success\n")
            for tid in sorted(generation_log):
                fp.write(f"{tid}\t{generation_log[tid]}\n")
        print(f"[INFO] Generation log written to {gen_path}")
        return

    mean_recall = sum(final_dataset["recall"]) / len(final_dataset)
    mean_precision = sum(final_dataset["precision"]) / len(final_dataset)

    if args.output_dir is None:
        rel_model_path = os.path.relpath(os.path.normpath(args.model_path), os.getcwd())
        out_dir = os.path.join("runs", rel_model_path)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    trec_path = os.path.join(out_dir, "result.trec")
    with open(trec_path, "w", encoding="utf-8") as trec:
        for topicid, doc_ids in zip(final_dataset["topicid"], final_dataset["retrieved_ids"]):
            for rank, doc_id in enumerate(doc_ids[:1000]):
                trec.write(f"{topicid} Q0 {doc_id} {rank + 1} {1000 - rank} run\n")

    def fscore(rec, prec, beta):
        if rec + prec == 0:
            return 0.0
        return (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)

    eval_path = os.path.join(out_dir, "result.eval")
    with open(eval_path, "w", encoding="utf-8") as ev:
        f1_list, f3_list, f5_list = [], [], []
        for tid, rec, prec in zip(final_dataset["topicid"], final_dataset["recall"], final_dataset["precision"]):
            f1 = fscore(rec, prec, 1)
            f3 = fscore(rec, prec, 3)
            f5 = fscore(rec, prec, 5)
            f1_list.append(f1)
            f3_list.append(f3)
            f5_list.append(f5)
            ev.write(f"{tid}\t{rec:.4f}\t{prec:.4f}\t{f1:.4f}\t{f3:.4f}\t{f5:.4f}\n")

        ev.write(
            f"AVG\t{mean_recall:.4f}\t{mean_precision:.4f}"
            f"\t{sum(f1_list)/len(f1_list):.4f}"
            f"\t{sum(f3_list)/len(f3_list):.4f}"
            f"\t{sum(f5_list)/len(f5_list):.4f}\n"
        )

    gen_path = os.path.join(out_dir, "generation_success_log.tsv")
    with open(gen_path, "w", encoding="utf-8") as fp:
        fp.write("topicid\tnum_generations_to_success\n")
        for tid in sorted(generation_log):
            fp.write(f"{tid}\t{generation_log[tid]}\n")
    jsonl_path = os.path.join(out_dir, "generated_responses.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jfp:
        for topicid, response, query in zip(
            final_dataset["topicid"], final_dataset["output"], final_dataset["boolean_query"]
        ):
            jfp.write(json.dumps({
                "topicid": topicid,
                "response": response,
                "boolean_query": query
            }, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved generated queries and responses to {jsonl_path}")

    print("\n==== Final Metrics ====")
    print(f"Recall   : {mean_recall:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Files saved under {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate GRPO model with optional vLLM")
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=4096)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--ref_date", action="store_true", help="Use reference dates for retrieval")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument(
        "--merge_only",
        action="store_true",
        help="Merge PEFT adapter into base model then exit.",
    )
    args = p.parse_args()
    set_seed(42)

    if args.merge_only:
        merge_peft_model_and_save(args.model_path)
    else:
        if not os.path.exists(f"{args.model_path}/config.json"):
            merge_peft_model_and_save(args.model_path)
        main(args)

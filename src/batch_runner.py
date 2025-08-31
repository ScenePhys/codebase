#!/usr/bin/env python3
import os
import sys
import csv
import json
import time
import argparse
import concurrent.futures as futures
from typing import Dict, Any, List, Optional, Tuple

import hashlib
import pandas as pd

# Reuse frame extraction from working_video_analysis
from working_video_analysis import extract_frames_b64, make_messages_with_frames
from openai import OpenAI

# Reuse evaluation metrics
from robust_evaluation import RobustPhysicsEvaluator

AVAL_OPENAI_BASE = "https://api.avalai.ir/v1"

MODEL_IDS = [
    ("gpt-4o-mini", "openai"),
    ("gemini-2.5-flash-lite", "gemini"),
    ("qwen-vl-plus", "qwen"),
]

# Heuristics for CSV schema
CANDIDATE_VIDEO_COLS = ["video", "video_id", "filename", "file", "path", "video_name", "video_filename"]
CANDIDATE_QTYPE_COLS = ["question_type", "type", "qtype"]
CANDIDATE_QUESTION_COLS = ["question", "prompt", "query"]
CANDIDATE_GOLD_COLS = ["gold", "answer", "gold_answer", "reference", "target"]

# PhET triplet columns
TRIPLET_SCHEMA = {
    "video": ["video_filename"],
    "q_conceptual": ["Q_conceptual"],
    "a_conceptual": ["A_conceptual"],
    "q_numerical": ["Q_numerical"],
    "a_numerical": ["A_numerical"],
    "q_error": ["Q_error_detection", "Q_error", "Q_err"],
    "a_error": ["A_error_detection", "A_error", "A_err"],
}


def stable_key(category: str, video_name: str, qtype: str, question: str) -> str:
    h = hashlib.md5(question.encode("utf-8")).hexdigest()[:12]
    return f"{category}__{video_name}__{qtype}__{h}"


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def map_video_path(video_dataset_dir: str, category: str, filename: str) -> Optional[str]:
    # Try direct join
    p = os.path.join(video_dataset_dir, category, filename)
    if os.path.exists(p):
        return p
    # Try adding .mp4 if missing
    if not filename.lower().endswith(".mp4"):
        p2 = os.path.join(video_dataset_dir, category, filename + ".mp4")
        if os.path.exists(p2):
            return p2
    # Try case-insensitive match within category
    cat_dir = os.path.join(video_dataset_dir, category)
    if os.path.isdir(cat_dir):
        fl = filename.lower()
        for f in os.listdir(cat_dir):
            if f.lower() == fl or f.lower() == fl + ".mp4":
                return os.path.join(cat_dir, f)
    return None


def call_vlm(client: OpenAI, model: str, frames_b64: List[str], question: str, temperature: float = 0.2, retries: int = 3, backoff: float = 1.5) -> str:
    messages = make_messages_with_frames(question, frames_b64)
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=600,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt >= retries:
                raise
            print(f"  Retry {attempt}/{retries} for {model} due to: {e}")
            time.sleep(backoff ** attempt)


def process_row(args: Tuple) -> Dict[str, Any]:
    (
        api_key, video_dataset_dir, category, video_name, question_type, question, gold_answer,
        fps, max_frames, jpg_quality, checkpoint_dir
    ) = args
    
    # Create stable checkpoint key
    ckpt_key = stable_key(category, video_name, question_type, question)
    ckpt_file = os.path.join(checkpoint_dir, ckpt_key + ".json")
    
    # Check if already completed successfully
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, 'r') as f:
                existing = json.load(f)
                if existing.get("ok") and "samples" in existing and len(existing["samples"]) == 3:
                    print(f"  Skipping {ckpt_key}: already completed successfully")
                    return existing
        except Exception:
            pass  # Corrupted checkpoint, will rerun
    
    print(f"  Processing {ckpt_key}")
    
    # Map video path
    video_path = map_video_path(video_dataset_dir, category, video_name)
    if not video_path:
        result = {
            "category": category,
            "video": video_name,
            "question_type": question_type,
            "question": question,
            "gold_answer": gold_answer,
            "ok": False,
            "error": f"Video file not found: {video_name}",
            "samples": []
        }
        # Save checkpoint even for failures
        with open(ckpt_file, 'w') as f:
            json.dump(result, f, indent=2)
        return result
    
    try:
        # Extract frames
        frames_b64 = extract_frames_b64(video_path, fps, max_frames, jpg_quality)
        if not frames_b64:
            raise Exception("No frames extracted")
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=AVAL_OPENAI_BASE,
        )
        
        # Call all VLMs
        samples = []
        for model_id, model_type in MODEL_IDS:
            try:
                print(f"    Calling {model_id}...")
                start_time = time.time()
                response = call_vlm(client, model_id, frames_b64, question)
                elapsed = time.time() - start_time
                
                # Evaluate response
                evaluator = RobustPhysicsEvaluator()
                eval_result = evaluator.evaluate_response(
                    video_id=os.path.basename(video_path),
                    topic=category,
                    question_type=question_type,
                    model_name=model_id,
                    question=question,
                    vlm_response=response,
                    ground_truth=gold_answer
                )
                
                samples.append({
                    "model": model_id,
                    "response": response,
                    "evaluation": eval_result,
                    "elapsed": elapsed,
                    "error": None
                })
                print(f"    ✓ {model_id} completed in {elapsed:.1f}s")
                
            except Exception as e:
                error_msg = str(e)
                print(f"    ✗ {model_id} failed: {error_msg}")
                samples.append({
                    "model": model_id,
                    "response": None,
                    "evaluation": None,
                    "elapsed": None,
                    "error": error_msg
                })
        
        # Check if all models failed
        successful_samples = [s for s in samples if s["error"] is None]
        if not successful_samples:
            result = {
                "category": category,
                "video": video_name,
                "question_type": question_type,
                "question": question,
                "gold_answer": gold_answer,
                "ok": False,
                "error": "All VLM calls failed",
                "samples": samples
            }
        else:
            result = {
                "category": category,
                "video": video_name,
                "question_type": question_type,
                "question": question,
                "gold_answer": gold_answer,
                "ok": True,
                "error": "",
                "samples": samples
            }
        
        # Save checkpoint
        with open(ckpt_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        result = {
            "category": category,
            "video": video_name,
            "question_type": question_type,
            "question": question,
            "gold_answer": gold_answer,
            "ok": False,
            "error": str(e),
            "samples": []
        }
        # Save checkpoint even for failures
        with open(ckpt_file, 'w') as f:
            json.dump(result, f, indent=2)
        return result


def expand_triplets_jobs(df: pd.DataFrame, category: str, api_key: str, video_dataset_dir: str,
                         fps: float, max_frames: int, jpg_quality: int, checkpoint_dir: str) -> List[Tuple]:
    jobs: List[Tuple] = []
    
    # Find columns using schema
    vcol = find_col(df, TRIPLET_SCHEMA["video"])
    qc_col = find_col(df, TRIPLET_SCHEMA["q_conceptual"])
    ac_col = find_col(df, TRIPLET_SCHEMA["a_conceptual"])
    qn_col = find_col(df, TRIPLET_SCHEMA["q_numerical"])
    an_col = find_col(df, TRIPLET_SCHEMA["a_numerical"])
    qe_col = find_col(df, TRIPLET_SCHEMA["q_error"])
    ae_col = find_col(df, TRIPLET_SCHEMA["a_error"])
    
    if not all([vcol, qc_col, ac_col, qn_col, an_col, qe_col, ae_col]):
        return jobs
    
    for _, row in df.iterrows():
        video = str(row[vcol]).strip()
        if not video or pd.isna(video):
            continue
            
        # Conceptual question
        if qc_col and ac_col and str(row[qc_col]).strip() and str(row[ac_col]).strip():
            jobs.append((api_key, video_dataset_dir, category, video, "conceptual",
                        str(row[qc_col]).strip(), str(row[ac_col]).strip(),
                        fps, max_frames, jpg_quality, checkpoint_dir))
        
        # Numerical question
        if qn_col and an_col and str(row[qn_col]).strip() and str(row[an_col]).strip():
            jobs.append((api_key, video_dataset_dir, category, video, "numerical",
                        str(row[qn_col]).strip(), str(row[an_col]).strip(),
                        fps, max_frames, jpg_quality, checkpoint_dir))
        
        # Error detection question
        if qe_col and ae_col and str(row[qe_col]).strip() and str(row[ae_col]).strip():
            jobs.append((api_key, video_dataset_dir, category, video, "error_detection",
                        str(row[qe_col]).strip(), str(row[ae_col]).strip(),
                        fps, max_frames, jpg_quality, checkpoint_dir))
    
    return jobs


def expand_generic_jobs(df: pd.DataFrame, category: str, api_key: str, video_dataset_dir: str,
                        fps: float, max_frames: int, jpg_quality: int, checkpoint_dir: str) -> List[Tuple]:
    jobs: List[Tuple] = []
    vcol = find_col(df, CANDIDATE_VIDEO_COLS)
    qcol = find_col(df, CANDIDATE_QUESTION_COLS)
    tcol = find_col(df, CANDIDATE_QTYPE_COLS)
    gcol = find_col(df, CANDIDATE_GOLD_COLS)
    if not all([vcol, qcol, tcol, gcol]):
        return jobs
    for _, row in df.iterrows():
        jobs.append((api_key, video_dataset_dir, category,
                    str(row[vcol]).strip(),
                    str(row[tcol]).strip().lower(),
                    str(row[qcol]).strip(),
                    str(row[gcol]).strip(),
                    fps, max_frames, jpg_quality, checkpoint_dir))
    return jobs


def load_existing_results(results_file: str) -> List[Dict[str, Any]]:
    """Load existing results and identify failed items"""
    if not os.path.exists(results_file):
        return []
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Identify failed items
        failed_items = []
        for item in results:
            if not item.get("ok") or "samples" not in item:
                failed_items.append(item)
            else:
                # Check if any samples have errors
                samples = item.get("samples", [])
                if any(s.get("error") for s in samples):
                    failed_items.append(item)
        
        print(f"Found {len(failed_items)} failed items in existing results")
        return failed_items
    except Exception as e:
        print(f"Error loading existing results: {e}")
        return []


def create_rerun_jobs(failed_items: List[Dict[str, Any]], api_key: str, video_dataset_dir: str,
                      fps: float, max_frames: int, jpg_quality: int, checkpoint_dir: str) -> List[Tuple]:
    """Convert failed items back to job tuples for rerunning"""
    jobs = []
    for item in failed_items:
        jobs.append((
            api_key, video_dataset_dir, item["category"], item["video"], 
            item["question_type"], item["question"], item["gold_answer"],
            fps, max_frames, jpg_quality, checkpoint_dir
        ))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Parallel batch evaluation for Physics Video QA")
    parser.add_argument("--video_dataset", required=True, help="Path to Video_dataset root")
    parser.add_argument("--metadata_dir", required=True, help="Path to metadata CSV folder")
    parser.add_argument("--api_key", default=os.environ.get("AVALAI_API_KEY", ""))
    parser.add_argument("--max_workers", type=int, default=4)  # Reduced default for stability
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--out", default="batch_results.json")
    parser.add_argument("--checkpoint_dir", default=".checkpoints")
    parser.add_argument("--rerun_failed", action="store_true", help="Rerun failed items from existing results")
    parser.add_argument("--existing_results", help="Path to existing results file for rerunning failed items")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Provide --api_key or set AVALAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.rerun_failed and args.existing_results:
        # Rerun failed items from existing results
        print(f"Loading failed items from {args.existing_results}")
        failed_items = load_existing_results(args.existing_results)
        if not failed_items:
            print("No failed items found to rerun")
            return
        
        jobs = create_rerun_jobs(failed_items, args.api_key, args.video_dataset,
                                args.fps, args.max_frames, args.jpg_quality, args.checkpoint_dir)
        print(f"Rerunning {len(jobs)} failed items...")
        
    else:
        # Normal run: discover CSVs and build jobs
        csv_files = sorted([f for f in os.listdir(args.metadata_dir) if f.lower().endswith(".csv")])
        if not csv_files:
            print("No CSV files found in metadata dir.", file=sys.stderr)
            sys.exit(1)

        jobs: List[Tuple] = []
        # Build tasks from each CSV
        for csv_name in csv_files:
            category = os.path.splitext(csv_name)[0]
            csv_path = os.path.join(args.metadata_dir, csv_name)
            df = pd.read_csv(csv_path)
            # First try PhET triplet schema
            triplet_jobs = expand_triplets_jobs(df, category, args.api_key, args.video_dataset, args.fps, args.max_frames, args.jpg_quality, args.checkpoint_dir)
            if triplet_jobs:
                jobs.extend(triplet_jobs)
            else:
                # Fall back to generic single Q/A per row
                gen_jobs = expand_generic_jobs(df, category, args.api_key, args.video_dataset, args.fps, args.max_frames, args.jpg_quality, args.checkpoint_dir)
                if gen_jobs:
                    jobs.extend(gen_jobs)
                else:
                    print(f"Skipping {csv_name}: could not detect schema (triplet or generic).")

        # Filter out jobs that already have successful checkpoints
        before = len(jobs)
        jobs = [j for j in jobs if not os.path.exists(os.path.join(args.checkpoint_dir, stable_key(j[2], j[3], j[4], j[5]) + ".json"))]
        after = len(jobs)

        print(f"Resuming: {before - after} already completed; dispatching {after} new question-items from {len(csv_files)} CSVs.")

    if not jobs:
        print("No jobs to process!")
        return

    print(f"Processing {len(jobs)} jobs with {args.max_workers} workers...")
    
    # Process jobs with better error handling
    completed = 0
    failed = 0
    
    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_job = {ex.submit(process_row, job): job for job in jobs}
        
        for future in futures.as_completed(future_to_job):
            try:
                result = future.result()
                if result.get("ok"):
                    completed += 1
                else:
                    failed += 1
                print(f"Progress: {completed + failed}/{len(jobs)} (✓{completed} ✗{failed})")
            except Exception as e:
                failed += 1
                job = future_to_job[future]
                print(f"Job failed with exception: {e}")
                print(f"  Job: {job[2]}/{job[3]}/{job[4]}")

    # Persist aggregate by reading all checkpoints (robust even if run interrupted)
    all_ckpts = []
    for fn in os.listdir(args.checkpoint_dir):
        if not fn.endswith('.json'):
            continue
        try:
            with open(os.path.join(args.checkpoint_dir, fn), 'r') as f:
                all_ckpts.append(json.load(f))
        except Exception:
            pass

    with open(args.out, "w") as f:
        json.dump(all_ckpts, f, indent=2)
    print(f"Wrote aggregated results to {args.out} (from {len(all_ckpts)} checkpoint items)")

    # Quick summary
    total = len(all_ckpts)
    ok = sum(1 for r in all_ckpts if r.get("ok"))
    print(f"Completed {ok}/{total} items successfully.")

if __name__ == "__main__":
    main()

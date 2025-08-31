#!/usr/bin/env python3
import os
import sys
import json
import re
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use the same AvalAI OpenAI-compatible endpoint and key as robust_evaluation.py
AVALAI_API_KEY = "aa-Fmk9AQbfxC1mZmEI0efUap2RfDpU7mLi67RIez5pmcUmZ7ym"
AVALAI_BASE_URL = "https://api.avalai.ir/v1"

try:
    import openai
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

JUDGE_MODEL = "gpt-4o-mini"  # pin if provider supports versioning
JUDGE_SYSTEM = "You are a strict, consistent physics grader. Output only JSON."

# ---------- Deterministic Numeric Scorer ----------

def score_numerical(pred_text: str, gold_value: float, tol_abs: float = None, tol_rel: float = 0.05, must_have_units: List[str] = None) -> Tuple[float, bool, str]:
    """
    Returns (score_0_to_5, passed_bool, notes).
    - Parse number from pred_text.
    - Check required units (string contains).
    - Compare to gold with abs/rel tolerance.
    """
    notes: List[str] = []
    # Number parse
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pred_text)
    if not m:
        return 1.0, False, "No numeric value found."
    try:
        pred = float(m.group(0))
    except Exception:
        return 1.0, False, "Could not parse numeric value."

    # Unit check
    unit_ok = True
    if must_have_units and len(must_have_units) > 0:
        unit_ok = any(u.lower() in pred_text.lower() for u in must_have_units)
        if not unit_ok:
            notes.append("Missing/incorrect units.")

    # Tolerance check
    if tol_abs is not None and abs(pred - gold_value) <= tol_abs:
        val_ok = True
    elif tol_rel is not None and gold_value != 0 and abs(pred - gold_value) <= tol_rel * abs(gold_value):
        val_ok = True
    else:
        val_ok = False
        notes.append(f"Outside tolerance (pred={pred}, gold={gold_value}).")

    # Scoring: units (0/1), value (0/4)
    pts = 0
    pts += 1 if unit_ok else 0
    if val_ok:
        pts += 4
    else:
        # partial credit: within 2x tolerance → +2; otherwise +0
        if tol_abs is not None and abs(pred - gold_value) <= 2 * tol_abs:
            pts += 2
        elif tol_rel is not None and gold_value != 0 and abs(pred - gold_value) <= 2 * tol_rel * abs(gold_value):
            pts += 2

    return float(pts), (val_ok and unit_ok), "; ".join(notes) if notes else "OK"

# ---------- JSON-only LLM Judge with rubric ----------

def build_rubric_prompt(question: str, response: str, question_type: str) -> str:
    rubric_map = {
        "conceptual": [
            "States correct qualitative relationship and directionality.",
            "Names and applies the governing law/principle correctly.",
            "Addresses conditions/assumptions; no major physics errors.",
            "Grounds answer in the clip (mentions on-screen values/objects) when relevant.",
            "Clear, concise explanation.",
        ],
        "error_detection": [
            "Identifies the most impactful idealization/limitation in the clip.",
            "Explains the physical consequence if violated (correct direction of change).",
            "No major physics errors; considers confounders if relevant.",
            "Grounds critique in visual evidence (gauges/sliders/geometry) when relevant.",
            "Clear, concise explanation.",
        ],
    }
    rubric = rubric_map.get(question_type, rubric_map["conceptual"])  # default to conceptual rubric

    return f"""
You will grade a {question_type} physics answer on a 0–5 integer scale using this checklist:
- {rubric[0]}
- {rubric[1]}
- {rubric[2]}
- {rubric[3]}
- {rubric[4]}

Scoring guide:
5 = all checklist items satisfied; 4 = one minor miss; 3 = some correct but with gaps; 2 = mostly incorrect; 1 = off-topic/wrong.
Return STRICT JSON ONLY (no prose) with fields:
{{
  "score": <int 1-5>,
  "reason": "<one-sentence justification>",
  "flags": ["units_issue" | "law_missing" | "direction_error" | "no_visual_grounding" | "other"...]
}}

Question: {question}
Answer: {response}
""".strip()


def call_json_judge(question: str, response: str, question_type: str, client: Any) -> Dict[str, Any]:
    prompt = build_rubric_prompt(question, response, question_type)
    out = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    ).choices[0].message.content
    try:
        js = json.loads(out)
        score = int(js.get("score", 3))
        score = max(1, min(5, score))
        reason = js.get("reason", "")
        flags = js.get("flags", [])
        if not isinstance(flags, list):
            flags = ["invalid_flags_format"]
        return {"score": float(score), "reason": reason, "flags": flags, "raw": js}
    except Exception:
        return {"score": 1.0, "reason": "Non-JSON judge output.", "flags": ["parse_error"], "raw": out}


def dual_judge(question: str, response: str, question_type: str) -> Dict[str, Any]:
    # Two independent passes (deterministic, temperature=0). If API supports seed, set different seeds; else same.
    client = openai.OpenAI(api_key=AVALAI_API_KEY, base_url=AVALAI_BASE_URL)
    j1 = call_json_judge(question, response, question_type, client)
    j2 = call_json_judge(question, response, question_type, client)
    avg = (j1["score"] + j2["score"]) / 2.0
    flags = sorted(list(set(j1.get("flags", []) + j2.get("flags", []))))
    return {"judge1": j1, "judge2": j2, "avg_score": avg, "flags": flags}

# ---------- Utilities ----------

UNIT_PATTERN = re.compile(r"\b(m/s|m/s²|kg|N|J|W|V|A|Ω|F|H|rad|deg|°|Pa|m|cm|mm|km|s)\b", re.IGNORECASE)
NUM_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_gold_number_and_units(gold_answer: str) -> Tuple[float, List[str]]:
    if not gold_answer:
        return None, []
    m = NUM_PATTERN.search(gold_answer)
    gold = float(m.group(0)) if m else None
    units = list(set(u.group(0) for u in UNIT_PATTERN.finditer(gold_answer)))
    return gold, units

# ---------- Main re-evaluation pipeline ----------

def process_item(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    category = item.get("category")
    question_type = item.get("question_type", "conceptual")
    question = item.get("question") or f"{question_type} question for {category}"
    gold_answer = item.get("gold_answer") or item.get("ground_truth")

    # Prepare numeric golds
    gold_val, gold_units = extract_gold_number_and_units(gold_answer or "")

    new_samples: List[Dict[str, Any]] = []
    for sample in item.get("samples", []):
        model = sample.get("model")
        answer = sample.get("answer") or sample.get("response") or ""
        if not answer or answer.startswith("Error occurred"):
            # Preserve structure, mark as unscored
            updated = dict(sample)
            updated.setdefault("metrics_standard", {})
            updated["metrics_standard"].update({
                "judge_avg": None,
                "judge": None,
                "numeric_score": None,
                "numeric_pass": None,
                "numeric_notes": "skipped/error",
            })
            new_samples.append(updated)
            continue

        updated = dict(sample)
        updated.setdefault("metrics_standard", {})

        if question_type == "numerical" and gold_val is not None:
            # Deterministic numeric scorer
            score, passed, notes = score_numerical(
                pred_text=answer,
                gold_value=gold_val,
                tol_abs=None,
                tol_rel=0.05,
                must_have_units=gold_units if gold_units else None,
            )
            updated["metrics_standard"].update({
                "numeric_score": score,
                "numeric_pass": passed,
                "numeric_notes": notes,
            })
            # No LLM judge for numerical
            updated["metrics_standard"].update({"judge_avg": None, "judge": None})
        else:
            # Conceptual / error-detection → JSON-only judge (dual)
            judge = dual_judge(question, answer, question_type)
            updated["metrics_standard"].update({
                "judge_avg": judge.get("avg_score"),
                "judge": judge,
                "numeric_score": None,
                "numeric_pass": None,
                "numeric_notes": None,
            })

        new_samples.append(updated)

    new_item = dict(item)
    new_item["samples"] = new_samples
    return new_item


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-evaluate with standard judge and numeric scorer")
    parser.add_argument("--results", default="batch_results.json", help="Input results JSON")
    parser.add_argument("--out", default="batch_result_new.json", help="Output results JSON")
    parser.add_argument("--max_workers", type=int, default=6, help="Parallel workers")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Input not found: {args.results}")
        sys.exit(1)

    with open(args.results, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} items. Starting standard evaluation with {args.max_workers} workers...")

    start = time.time()
    out_items: List[Dict[str, Any]] = [None] * len(results)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process_item, i, item): i for i, item in enumerate(results)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                out_items[i] = fut.result()
            except Exception as e:
                print(f"Item {i} failed: {e}")
                out_items[i] = results[i]
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"Processed {i + 1}/{len(results)} in {elapsed/60:.1f} min")

    with open(args.out, "w") as f:
        json.dump(out_items, f, indent=2)

    elapsed = time.time() - start
    print(f"Done. Wrote {args.out}. Total time {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()

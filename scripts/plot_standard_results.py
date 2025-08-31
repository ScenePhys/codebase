#!/usr/bin/env python3
import os
import sys
import json
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import pandas as pd

# Prefer plotly; fall back to matplotlib if needed
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTLY = False

INPUT_JSON = "batch_result_new.json"
OUT_DIR = "plots"

# Map lab/category names to general fields (based on provided sunburst)
GENERAL_FIELD_MAP: Dict[str, str] = {
    # Mechanics & Fluids
    "Buoyancy": "Mechanics & Fluids",
    "Projectile": "Mechanics & Fluids",
    "Collision": "Mechanics & Fluids",
    "Masses & Springs": "Mechanics & Fluids",
    "Pendulum": "Mechanics & Fluids",
    # Optics
    "Flat Mirror": "Optics",
    "Concave Lens": "Optics",
    "Convex Mirror": "Optics",
    "Convex Lens": "Optics",
    "Concave Mirror": "Optics",
    # Electromagnetism & Circuits
    "Generator": "Electromagnetism & Circuits",
    "Timeconstant": "Electromagnetism & Circuits",
    "Coulomb": "Electromagnetism & Circuits",
    "Capacitance": "Electromagnetism & Circuits",
    # Quantum / Modern Physics
    "Photon": "Quantum / Modern Physics",
    "Quantum": "Quantum / Modern Physics",
    "Hydrogen": "Quantum / Modern Physics",
}

DIFFICULTY_ORDER = ["easy", "moderate", "hard"]
QTYPE_ORDER = ["conceptual", "error_detection", "numerical"]


def load_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Input not found: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def compute_sample_score(sample: Dict[str, Any], question_type: str) -> float:
    # Standard metrics structure set in rerun_standard_judge.py
    ms = (sample or {}).get("metrics_standard") or {}
    if question_type == "numerical":
        num = ms.get("numeric_score")
        if num is None:
            return None
        # numeric_score is 0..5 already
        return float(num)
    # conceptual/error-detection use judge avg 1..5
    j = ms.get("judge_avg")
    return float(j) if j is not None else None


def build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in results:
        lab = item.get("category") or item.get("lab") or "Unknown"
        general = GENERAL_FIELD_MAP.get(lab, "Other")
        qtype = item.get("question_type", "conceptual")
        diff = (item.get("difficulty") or "unknown").lower()
        for s in item.get("samples", []):
            model = s.get("model") or "unknown"
            score = compute_sample_score(s, qtype)
            if score is None:
                continue
            rows.append({
                "lab": lab,
                "field": general,
                "question_type": qtype,
                "difficulty": diff,
                "model": model,
                "score": score,
            })
    if not rows:
        print("No rows with scores found in input.")
    return pd.DataFrame(rows)


def ensure_outdir():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)


def plot_grouped_bar(df: pd.DataFrame, x_col: str, hue_col: str, title: str, filename: str):
    ensure_outdir()
    path_png = os.path.join(OUT_DIR, filename + ".png")
    path_html = os.path.join(OUT_DIR, filename + ".html")
    if PLOTLY:
        fig = px.bar(df, x=x_col, y="score", color=hue_col, barmode="group",
                     category_orders={
                         "difficulty": DIFFICULTY_ORDER,
                         "question_type": QTYPE_ORDER
                     },
                     title=title)
        # Save HTML always; PNG if kaleido available
        try:
            fig.write_image(path_png, scale=2)
        except Exception:
            pass
        fig.write_html(path_html, include_plotlyjs="cdn")
    else:
        import matplotlib.pyplot as plt
        import numpy as np
        # Pivot for grouped bars
        pivot = df.pivot_table(index=x_col, columns=hue_col, values="score", aggfunc="mean")
        pivot = pivot.fillna(0)
        labels = list(pivot.index)
        groups = list(pivot.columns)
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(groups))
        fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.6), 6))
        for i, g in enumerate(groups):
            ax.bar(x + i*width, pivot[g].values, width, label=g)
        ax.set_xticks(x + (len(groups)-1)*width/2)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Mean score (0–5)")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_png, dpi=200)
        plt.close()


def generate_all_plots(df: pd.DataFrame):
    # Aggregate to mean score
    # 1) X -> lab (17), Y -> three bars Easy/Moderate/Hard, per model
    labs = sorted(df["lab"].unique())
    models = sorted(df["model"].unique())

    for model in models:
        dfm = df[df["model"] == model]
        agg1 = dfm.groupby(["lab", "difficulty"], as_index=False)["score"].mean()
        plot_grouped_bar(
            agg1, x_col="lab", hue_col="difficulty",
            title=f"Mean score by Lab and Difficulty — {model}",
            filename=f"case1_lab_difficulty_{model}"
        )

        # 2) X -> general fields (4), Y -> three bars Easy/Moderate/Hard
        agg2 = dfm.groupby(["field", "difficulty"], as_index=False)["score"].mean()
        plot_grouped_bar(
            agg2, x_col="field", hue_col="difficulty",
            title=f"Mean score by Field and Difficulty — {model}",
            filename=f"case2_field_difficulty_{model}"
        )

        # 3) X -> lab, Y -> three bars by question type
        agg3 = dfm.groupby(["lab", "question_type"], as_index=False)["score"].mean()
        plot_grouped_bar(
            agg3, x_col="lab", hue_col="question_type",
            title=f"Mean score by Lab and Question Type — {model}",
            filename=f"case3_lab_qtype_{model}"
        )

        # 4) X -> field, Y -> three bars by question type
        agg4 = dfm.groupby(["field", "question_type"], as_index=False)["score"].mean()
        plot_grouped_bar(
            agg4, x_col="field", hue_col="question_type",
            title=f"Mean score by Field and Question Type — {model}",
            filename=f"case4_field_qtype_{model}"
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate grouped bar charts from batch_result_new.json")
    parser.add_argument("--input", default=INPUT_JSON, help="Path to batch_result_new.json")
    args = parser.parse_args()

    results = load_results(args.input)
    df = build_dataframe(results)
    if df.empty:
        print("No data to plot.")
        sys.exit(0)
    generate_all_plots(df)
    print(f"Saved plots to {OUT_DIR}/ (HTML and PNG when available)")

if __name__ == "__main__":
    main()

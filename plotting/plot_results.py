"""Plot per-category scores across harnesses from runner.py result JSONs.

Walks a results tree (<results_dir>/<model>/<dataset>/<strategy>/*.json), reduces each
category file to one scalar (MC accuracy / counting partial credit / mean judge score if a
*_judge_*.json sidecar exists, else ROUGE-L), and draws one grouped bar chart per
model+dataset: categories on the x-axis, one bar per harness.

    python -m plotting.plot_results results/ --out figures/
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGY_ORDER = ["uniform", "stitched", "decentralized"]


def score_category_file(path):
    """One scalar in [0,1] for a per-category results JSON (same reduction as runner tables)."""
    # prefer the LLM-judge sidecar for summarization if it exists
    for mode in ("rts", "claims"):
        sidecar = os.path.splitext(path)[0] + f"_judge_{mode}.json"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                return json.load(f).get("mean_score", 0.0)
    with open(path) as f:
        data = json.load(f)
    if not data:
        return None
    entries = list(data.values())
    if "rouge_scores" in entries[0]:
        return sum(e["rouge_scores"]["rougeL"] for e in entries) / len(entries)
    if "partial_credit" in entries[0]:
        return sum(e["partial_credit"] for e in entries) / len(entries)
    return sum(e.get("is_correct", 0) for e in entries) / len(entries)


def collect(results_dir):
    """{(model, dataset): {category: {strategy: score}}}"""
    table = defaultdict(lambda: defaultdict(dict))
    pattern = os.path.join(results_dir, "*", "*", "*", "*.json")
    for path in sorted(glob.glob(pattern)):
        if "_judge_" in os.path.basename(path):
            continue
        model, dataset, strategy = path.split(os.sep)[-4:-1]
        # filename: <model>_<dataset>_<strategy>_<category>[_passN].json
        stem = os.path.splitext(os.path.basename(path))[0]
        category = stem.split(f"_{strategy}_", 1)[-1]
        score = score_category_file(path)
        if score is not None:
            table[(model, dataset)][category][strategy] = score
    return table


def plot(table, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for (model, dataset), cats in table.items():
        categories = sorted(cats)
        strategies = [s for s in STRATEGY_ORDER if any(s in cats[c] for c in categories)]
        if not strategies:
            continue
        width = 0.8 / len(strategies)
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(categories)), 4))
        for si, strat in enumerate(strategies):
            xs = [ci + si * width for ci in range(len(categories))]
            ys = [cats[c].get(strat, 0.0) for c in categories]
            bars = ax.bar(xs, ys, width, label=strat)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks([ci + width * (len(strategies) - 1) / 2 for ci in range(len(categories))])
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(f"{model} — {dataset} (per-category, by harness)")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{model}_{dataset}_harness_comparison.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="root of runner.py outputs (<model>/<dataset>/<strategy>/)")
    parser.add_argument("--out", default="figures", help="directory for the PNGs")
    args = parser.parse_args()
    table = collect(args.results_dir)
    if not table:
        print(f"no result JSONs found under {args.results_dir}")
        return
    plot(table, args.out)


if __name__ == "__main__":
    main()

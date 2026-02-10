"""
Leaderboard Generation Script for Nasong

This script scans the `trained_models` directory and aggregates results from:
- config.yaml (Training configuration)
- history.json (Training loss history)
- evaluation.json (Note detection evaluation)

It generates a markdown report `results_analysis/leaderboards.md`.

Usage:
    nasong-leaderboard [options]
"""

#
### Import Modules. ###
#
from typing import Dict, Any, List

#
import os
import json
import yaml
import argparse
import pandas as pd


def calculate_note_score(
    target_notes: List[Dict], predicted_notes: List[Dict], tolerance: float = 0.05
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-score for detected notes.
    Matching is based on frequency overlap within tolerance.
    """
    if not target_notes and not predicted_notes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not target_notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not predicted_notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_targets = set()
    matched_preds = set()

    # Simple matching logic: overlapping frequency
    for i, t_note in enumerate(target_notes):
        t_freqs = t_note.get("frequencies", [])
        if not t_freqs:
            continue
        t_mean = sum(t_freqs) / len(t_freqs)

        for j, p_note in enumerate(predicted_notes):
            if j in matched_preds:
                continue

            p_freqs = p_note.get("frequencies", [])
            if not p_freqs:
                continue
            p_mean = sum(p_freqs) / len(p_freqs)

            # Check if within tolerance
            if abs(t_mean - p_mean) / t_mean <= tolerance:
                matched_targets.add(i)
                matched_preds.add(j)
                break

    tp = len(matched_targets)
    fp = len(predicted_notes) - len(matched_preds)
    fn = len(target_notes) - len(matched_targets)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def load_experiment_data(exp_dir: str) -> Dict[str, Any]:
    """Load all relevant data for an experiment."""
    data = {"name": os.path.basename(exp_dir)}

    # 1. Config
    config_path = os.path.join(exp_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            data["instrument"] = config.get("instrument_name", "unknown")
            data["epochs"] = config.get("epochs", 0)
            data["lr"] = config.get("learning_rate", 0.0)
            data["method"] = config.get("note_detection", {}).get("method", "unknown")
            data["device"] = config.get("device", "unknown")
            data["train_dur"] = config.get("train_duration", 0)
    else:
        # Try to guess from other files or skip
        data["instrument"] = "unknown"
        data["method"] = "unknown"

    # 2. History
    history_path = os.path.join(exp_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            losses = history.get("losses", [])
            data["final_loss"] = losses[-1] if losses else None
            data["min_loss"] = min(losses) if losses else None
    else:
        data["final_loss"] = None
        data["min_loss"] = None

    # 3. Evaluation
    eval_path = os.path.join(exp_dir, "evaluation.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            evaluation = json.load(f)

            # Handle new 'splits' structure
            splits = evaluation.get("splits", {})
            results = {}

            # Priority: test > val > train > legacy 'results'
            if "test" in splits:
                results = splits["test"]
            elif "val" in splits:
                results = splits["val"]
            elif "train" in splits:
                results = splits["train"]
            elif "results" in evaluation:
                # Fallback to old format
                results = evaluation.get("results", {})

            target_res = results.get("target", {})
            trained_res = results.get("trained", {})

            # Use the experiment's method for primary metric if known, else first available
            method = data.get("method", "legacy")

            # If method specific results missing, finding first available that succeeded
            if (
                method not in target_res
                or target_res.get(method, {}).get("status") != "success"
            ):
                for m, res in target_res.items():
                    if res.get("status") == "success":
                        method = m
                        break

            data["eval_method"] = method

            if method and method in target_res and method in trained_res:
                t_data = target_res.get(method, {})
                p_data = trained_res.get(method, {})

                t_notes = t_data.get("notes", [])
                p_notes = p_data.get("notes", [])

                data["target_notes"] = len(t_notes)
                data["trained_notes"] = len(p_notes)

                # Calculate advanced scores
                scores = calculate_note_score(t_notes, p_notes)
                data["precision"] = scores["precision"]
                data["recall"] = scores["recall"]
                data["f1_score"] = scores["f1"]

    return data


def generate_markdown(experiments: List[Dict[str, Any]], output_path: str):
    """Generate markdown table."""

    df = pd.DataFrame(experiments)

    # Clean up floats
    for col in ["final_loss", "min_loss"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    for col in ["precision", "recall", "f1_score"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")

    content = "# 🏆 Nasong Experiment Leaderboard\n\n"
    content += "Aggregated results from `trained_models`. Evaluation prioritized on Test split.\n\n"

    # 1. Main Table
    content += "## All Experiments\n\n"

    cols = [
        "name",
        "instrument",
        "method",
        "eval_method",
        "epochs",
        "final_loss",
        "f1_score",
        "precision",
        "recall",
        "target_notes",
        "trained_notes",
    ]
    # Filter cols that exist
    cols = [c for c in cols if c in df.columns]

    # Sort by F1 (descending) if available
    if "f1_score" in df.columns:
        # To sort correctly we need numeric values, but we converted to string %
        # So we rely on the original order or re-sort before string conv.
        # For now, let's just let it be, or we could sort the list of dicts before DF creation.
        pass

    content += df[cols].to_markdown(index=False)
    content += "\n\n"

    # 2. Sub-leaderboards by Instrument
    content += "## By Instrument\n\n"
    if "instrument" in df.columns:
        for instrument, group in df.groupby("instrument"):
            content += f"### {instrument.capitalize()}\n\n"
            content += group[cols].to_markdown(index=False)
            content += "\n\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated leaderboard at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Nasong Leaderboard")
    parser.add_argument(
        "--models-dir", default="trained_models", help="Models directory"
    )
    parser.add_argument(
        "--output",
        default="results_analysis/leaderboards.md",
        help="Output markdown file",
    )

    args = parser.parse_args()

    experiments = []
    if os.path.exists(args.models_dir):
        # Walk through the directory to find experiments
        # An experiment is identified by the presence of config.yaml or history.json
        for root, dirs, files in os.walk(args.models_dir):
            if "config.yaml" in files or "history.json" in files:
                data = load_experiment_data(root)
                experiments.append(data)

    if not experiments:
        print("No experiments found.")
        return

    generate_markdown(experiments, args.output)


if __name__ == "__main__":
    main()


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

import os
import glob
import json
import yaml
import argparse
from typing import Dict, Any, List
import pandas as pd

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
            results = evaluation.get("results", {})
            target_res = results.get("target", {})
            trained_res = results.get("trained", {})

            # Use the experiment's method for primary metric if known, else first available
            method = data.get("method", "legacy")
            if method not in target_res:
                # Fallback
                method = list(target_res.keys())[0] if target_res else None

            if method:
                t_count = target_res.get(method, {}).get("note_count", 0)
                p_count = trained_res.get(method, {}).get("note_count", 0)
                data["target_notes"] = t_count
                data["trained_notes"] = p_count
                data["note_diff"] = p_count - t_count
                data["note_accuracy"] = 1.0 - (abs(p_count - t_count) / max(1, t_count))

    return data

def generate_markdown(experiments: List[Dict[str, Any]], output_path: str):
    """Generate markdown table."""

    df = pd.DataFrame(experiments)

    # Clean up
    if "final_loss" in df.columns:
        df["final_loss"] = df["final_loss"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    if "note_accuracy" in df.columns:
        df["note_accuracy"] = df["note_accuracy"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")

    content = "# 🏆 Nasong Experiment Leaderboard\n\n"
    content += "Aggregated results from `trained_models`.\n\n"

    # 1. Main Table
    content += "## All Experiments\n\n"

    cols = ["name", "instrument", "method", "epochs", "final_loss", "target_notes", "trained_notes", "note_accuracy"]
    # Filter cols that exist
    cols = [c for c in cols if c in df.columns]

    # Sort by accuracy (descending) if available
    if "note_accuracy" in df.columns:
        # Convert back to numeric for sort? No, string now.
        # Re-sort using original list before formatting would be better but simple string sort might work for %
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
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Generated leaderboard at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Nasong Leaderboard")
    parser.add_argument("--models-dir", default="trained_models", help="Models directory")
    parser.add_argument("--output", default="results_analysis/leaderboards.md", help="Output markdown file")

    args = parser.parse_args()

    experiments = []
    if os.path.exists(args.models_dir):
        for d in os.listdir(args.models_dir):
            path = os.path.join(args.models_dir, d)
            if os.path.isdir(path):
                data = load_experiment_data(path)
                experiments.append(data)

    if not experiments:
        print("No experiments found.")
        return

    generate_markdown(experiments, args.output)

if __name__ == "__main__":
    main()

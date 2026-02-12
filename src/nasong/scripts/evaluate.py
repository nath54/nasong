"""
Evaluation and Visualization Script for Nasong

This script evaluates the performance of trained models by running various note detection
algorithms on both the target audio and the synthesized output. It generates:
1. A JSON report (`evaluation.json`) containing detected notes from all methods.
2. A spectogram comparison plot (`comparison_<instrument>.png`).

Usage:
    python -m nasong.scripts.evaluate [options]

Options:
    --models-dir DIR      Directory containing experiment folders (default: trained_models)
    --experiment NAME     Run only a specific experiment name (optional)
    --target-wav FILE     Run evaluation on a specific target WAV file (requires --trained-wav)
    --trained-wav FILE    Run evaluation on a specific trained WAV file (requires --target-wav)
    --output-dir DIR      Output directory for results (default: same as input or experiment dir)
    --methods LIST        Comma-separated list of methods to test (default: all)
"""

#
### Import Modules. ###
#
from typing import Any, Optional

#
import os
import glob
import json
import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

#
### Nasong imports ###
#
try:
    #
    from nasong.trainable.config import NoteDetectionConfig
    from nasong.trainable.note_detection.create import create_note_detector
#
except ImportError:
    #
    ### Fallback if running as script without package installed ###
    #
    import sys

    #
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../..", "src"))

    #
    from nasong.trainable.config import NoteDetectionConfig
    from nasong.trainable.note_detection.create import create_note_detector


#
ALL_METHODS = ["legacy", "basic_pitch", "librosa", "torchcrepe", "audioflux"]


def evaluate_audio(audio_path: str, methods: list[str] = None) -> dict[str, Any]:
    """Runs specified note detectors on an audio file.

    Args:
        audio_path (str): Path to the source WAV file.
        methods (list[str], optional): List of detection method names to run.
            Defaults to all supported methods.

    Returns:
        dict[str, Any]: A dictionary mapping method names to their results,
            including detected notes and success status.
    """
    if methods is None:
        methods = ALL_METHODS

    print(f"   Running detectors on {os.path.basename(audio_path)}...")

    results = {}

    try:
        audio, file_sr = sf.read(audio_path)
    except Exception as e:  # pylint: disable=broad-except
        print(f"   ⚠️ Failed to read audio: {e}")
        return {"error": str(e)}

    # Ensure float32 and mono
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            # normalized if int-based read wasn't automatic, but sf.read usually float64/32
            pass

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    for method in methods:
        try:
            # Create config
            config = NoteDetectionConfig(method=method)
            if method == "torchcrepe":
                config.crepe_model = "full"

            # Create detector
            detector = create_note_detector(config)

            # Run detection
            notes = detector.detect(audio, file_sr)

            # Serialize
            serializable_notes = []
            for note in notes:
                s_note = {}
                for k, v in note.items():
                    if isinstance(v, (np.floating, float)):
                        s_note[k] = float(v)
                    elif isinstance(v, (np.integer, int)):
                        s_note[k] = int(v)
                    elif isinstance(v, list):
                        s_note[k] = [
                            float(x) if isinstance(x, (np.floating, float)) else x
                            for x in v
                        ]
                    else:
                        s_note[k] = v
                serializable_notes.append(s_note)

            results[method] = {
                "status": "success",
                "note_count": len(serializable_notes),
                "notes": serializable_notes,
            }
            print(f"     ✅ {method}: {len(serializable_notes)} notes")

        except Exception as e:  # pylint: disable=broad-except
            results[method] = {"status": "failed", "error": str(e)}
            # print(f"     ❌ {method} failed: {e}")

    return results


def visualize_spectrograms(
    target_path: str,
    trained_path: str,
    output_dir: str,
    instrument_name: str,
    split_name: str = "train",
) -> None:
    """Generates and saves a side-by-side spectrogram comparison.

    Creates a vertical plot showing the target audio spectrogram above the
    synthesized audio spectrogram for visual quality assessment.

    Args:
        target_path (str): Path to the original target WAV.
        trained_path (str): Path to the rendered MIDI-to-audio WAV.
        output_dir (str): Directory where the PNG plot will be saved.
        instrument_name (str): The name of the instrument for the plot title.
        split_name (str, optional): Dataset split identifier (e.g., 'val').
            Defaults to "train".
    """
    if not os.path.exists(target_path) or not os.path.exists(trained_path):
        return

    target_audio, sr = sf.read(target_path)
    trained_audio, sr2 = sf.read(trained_path)

    if len(target_audio.shape) > 1:
        target_audio = target_audio.mean(axis=1)
    if len(trained_audio.shape) > 1:
        trained_audio = trained_audio.mean(axis=1)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.specgram(target_audio, Fs=sr, NFFT=2048, noverlap=512, cmap="inferno")
    plt.title(f"Target Spectrogram ({instrument_name})")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(2, 1, 2)
    plt.specgram(trained_audio, Fs=sr2, NFFT=2048, noverlap=512, cmap="inferno")
    plt.title(f"Synthesized Spectrogram ({instrument_name})")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    save_path = os.path.join(
        output_dir, f"comparison_{instrument_name}_{split_name}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_experiment(
    exp_dir: str, output_dir: Optional[str] = None, methods: list[str] = None
) -> None:
    """Evaluates an experiment directory across all dataset splits.

    Searches the experiment folder for train, val, and test audio pairs,
    runs note detection on each pair, and generates a unified JSON report
    and comparison plots.

    Args:
        exp_dir (str): Path to the experiment folder.
        output_dir (str, optional): Directory for evaluation output.
            Defaults to the experiment directory.
        methods (list[str], optional): Detection methods to test.
            Defaults to all methods.
    """
    print(f"\n📂 Processing {os.path.basename(exp_dir)}...")

    if output_dir is None:
        output_dir = exp_dir
    os.makedirs(output_dir, exist_ok=True)

    # Detect instrument name
    instrument_name = None

    # 1. Try reading from config.yaml
    config_path = os.path.join(exp_dir, "config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                instrument_name = config.get("instrument_name")
        except Exception as e:  # pylint: disable=broad-except
            print(f"   ⚠️ Failed to read config.yaml: {e}")

    # 2. Fallback to filename parsing
    if not instrument_name:
        train_targets = glob.glob(os.path.join(exp_dir, "*_trained_target.wav"))
        if not train_targets:
            # Fallback to legacy naming if exists
            train_targets = glob.glob(os.path.join(exp_dir, "*_target.wav"))
            if not train_targets:
                print(f"   ⚠️ No target audio found in {exp_dir}")
                return
            # Legacy: just take the bit before _target
            candidates = [
                os.path.basename(t).replace("_target.wav", "") for t in train_targets
            ]
        else:
            # Standard: take the bit before _trained_target
            candidates = [
                os.path.basename(t).replace("_trained_target.wav", "")
                for t in train_targets
            ]

        # Filter out prefixes like "val_" or "test_" if they exist
        clean_candidates = []
        for c in candidates:
            name = c
            if name.startswith("val_"):
                name = name[4:]
            elif name.startswith("test_"):
                name = name[5:]
            clean_candidates.append(name)

        # Pick the shortest candidate as it's likely the base instrument name
        if clean_candidates:
            instrument_name = min(clean_candidates, key=len)
        else:
            instrument_name = "unknown"

    splits = [
        ("train", "trained", "trained_target"),
        ("val", "val_trained", "val_trained_target"),
        ("test", "test_trained", "test_trained_target"),
    ]

    all_split_results = {}

    for split_key, trained_suffix, target_suffix in splits:
        target_path = os.path.join(exp_dir, f"{instrument_name}_{target_suffix}.wav")
        # Support various target naming conventions
        if not os.path.exists(target_path):
            if split_key == "train":
                target_path = os.path.join(exp_dir, f"{instrument_name}_target.wav")
            elif split_key == "val":
                target_path = os.path.join(exp_dir, f"{instrument_name}_val_target.wav")
            elif split_key == "test":
                target_path = os.path.join(
                    exp_dir, f"{instrument_name}_test_target.wav"
                )

        trained_path = os.path.join(exp_dir, f"{instrument_name}_{trained_suffix}.wav")

        if os.path.exists(target_path) and os.path.exists(trained_path):
            print(f"   📊 Split: {split_key}")

            # 1. Evaluate
            print("     🔍 Evaluating Target...")
            target_res = evaluate_audio(target_path, methods)

            print("     🔍 Evaluating Trained...")
            trained_res = evaluate_audio(trained_path, methods)

            all_split_results[split_key] = {
                "target": target_res,
                "trained": trained_res,
                "target_file": os.path.basename(target_path),
                "trained_file": os.path.basename(trained_path),
            }

            # 2. Viz
            visualize_spectrograms(
                target_path,
                trained_path,
                output_dir,
                instrument_name,
                split_name=split_key,
            )
            print(f"     ✅ Saved spectrogram comparison for {split_key}")
        else:
            if split_key == "train":
                print(f"   ⚠️ Missing train files: {target_path} or {trained_path}")

    # 3. Save JSON
    if all_split_results:
        evaluation_data = {
            "experiment": os.path.basename(exp_dir),
            "instrument": instrument_name,
            "splits": all_split_results,
        }

        json_path = os.path.join(output_dir, "evaluation.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2)
        print(f"   ✅ Saved full evaluation to {json_path}")


def main() -> None:
    """Main entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize Nasong training results."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="trained_models",
        help="Directory containing experiment folders",
    )
    parser.add_argument(
        "--experiment", type=str, help="Specific experiment name to run"
    )
    parser.add_argument("--target-wav", type=str, help="Specific target WAV file")
    parser.add_argument("--trained-wav", type=str, help="Specific trained WAV file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--methods", type=str, help="Comma-separated list of methods")

    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods else ALL_METHODS

    # Mode 1: Specific Files
    if args.target_wav and args.trained_wav:
        out_dir = (
            args.output_dir if args.output_dir else os.path.dirname(args.trained_wav)
        )
        instrument_name = (
            os.path.basename(args.target_wav).replace("_target.wav", "") or "unknown"
        )

        print("📂 Processing specific files...")
        target_results = evaluate_audio(args.target_wav, methods)
        trained_results = evaluate_audio(args.trained_wav, methods)

        evaluation_data = {
            "experiment": "custom",
            "instrument": instrument_name,
            "target_audio": args.target_wav,
            "trained_audio": args.trained_wav,
            "results": {"target": target_results, "trained": trained_results},
        }

        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "evaluation.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2)
        print(f"   ✅ Saved evaluation to {json_path}")

        visualize_spectrograms(
            args.target_wav, args.trained_wav, out_dir, instrument_name
        )
        return

    # Mode 2: Iterate Experiments
    experiments = []
    if args.experiment:
        # If specific name provided, we might ideally search for it,
        # but for now let's assume it's just a filter if we find it in traversal or direct path
        # Actually simplest is to just traverse and filter.
        pass

    if os.path.exists(args.models_dir):
        # Walk through the directory to find experiments similar to leaderboard.py
        for root, dirs, files in os.walk(args.models_dir):
            if "config.yaml" in files or "history.json" in files:
                experiments.append(root)
    else:
        print(f"❌ Models directory not found: {args.models_dir}")
        return

    print(f"Found {len(experiments)} experiments in {args.models_dir}.")

    for exp_dir in experiments:
        # exp_dir is now the full path from os.walk

        # Filter if specific experiment name requested (check if name is part of path)
        if args.experiment and args.experiment not in exp_dir:
            continue

        try:
            process_experiment(exp_dir, args.output_dir, methods)
        except Exception as e:  # pylint: disable=broad-except
            print(f"   ❌ Error processing {exp_dir}: {e}")


if __name__ == "__main__":
    main()


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

import os
import glob
import json
import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

# Nasong imports
try:
    from nasong.trainable.config import NoteDetectionConfig
    from nasong.trainable.note_detection.create import create_note_detector
except ImportError:
    # Fallback if running as script without package installed
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../..", "src"))
    from nasong.trainable.config import NoteDetectionConfig
    from nasong.trainable.note_detection.create import create_note_detector

ALL_METHODS = [
    "legacy",
    "basic_pitch",
    "librosa",
    "torchcrepe",
    "audioflux"
]

def evaluate_audio(audio_path: str, methods: List[str] = None) -> Dict[str, Any]:
    """
    Run specified note detectors on the given audio.
    """
    if methods is None:
        methods = ALL_METHODS

    print(f"   Running detectors on {os.path.basename(audio_path)}...")

    results = {}

    try:
        audio, file_sr = sf.read(audio_path)
    except Exception as e:
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
            if method == 'torchcrepe':
                config.crepe_model = 'full'

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
                        s_note[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                    else:
                        s_note[k] = v
                serializable_notes.append(s_note)

            results[method] = {
                "status": "success",
                "note_count": len(serializable_notes),
                "notes": serializable_notes
            }
            print(f"     ✅ {method}: {len(serializable_notes)} notes")

        except Exception as e:
            results[method] = {
                "status": "failed",
                "error": str(e)
            }
            # print(f"     ❌ {method} failed: {e}")

    return results

def visualize_spectrograms(target_path, trained_path, output_dir, instrument_name, split_name: str = "train"):
    """
    Generate and save a side-by-side spectrogram comparison.
    """
    if not os.path.exists(target_path) or not os.path.exists(trained_path):
        return

    target_audio, sr = sf.read(target_path)
    trained_audio, sr2 = sf.read(trained_path)

    if len(target_audio.shape) > 1: target_audio = target_audio.mean(axis=1)
    if len(trained_audio.shape) > 1: trained_audio = trained_audio.mean(axis=1)

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.specgram(target_audio, Fs=sr, NFFT=2048, noverlap=512, cmap='inferno')
    plt.title(f"Target Spectrogram ({instrument_name})")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(2, 1, 2)
    plt.specgram(trained_audio, Fs=sr2, NFFT=2048, noverlap=512, cmap='inferno')
    plt.title(f"Synthesized Spectrogram ({instrument_name})")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    save_path = os.path.join(output_dir, f"comparison_{instrument_name}_{split_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_experiment(exp_dir: str, output_dir: Optional[str] = None, methods: List[str] = None):
    """
    Evaluate a single experiment directory across all available splits (train, val, test).
    """
    print(f"\n📂 Processing {os.path.basename(exp_dir)}...")

    if output_dir is None:
        output_dir = exp_dir
    os.makedirs(output_dir, exist_ok=True)

    # Detect instrument name from train target
    train_targets = glob.glob(os.path.join(exp_dir, "*_trained_target.wav"))
    if not train_targets:
        # Fallback to legacy naming if exists
        train_targets = glob.glob(os.path.join(exp_dir, "*_target.wav"))
        if not train_targets:
            print(f"   ⚠️ No target audio found in {exp_dir}")
            return
        instrument_name = os.path.basename(train_targets[0]).replace("_target.wav", "")
    else:
        instrument_name = os.path.basename(train_targets[0]).replace("_trained_target.wav", "")

    splits = [
        ("train", "trained", "trained_target"),
        ("val", "val_trained", "val_trained_target"),
        ("test", "test_trained", "test_trained_target")
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
                target_path = os.path.join(exp_dir, f"{instrument_name}_test_target.wav")
             
        trained_path = os.path.join(exp_dir, f"{instrument_name}_{trained_suffix}.wav")

        if os.path.exists(target_path) and os.path.exists(trained_path):
            print(f"   📊 Split: {split_key}")
            
            # 1. Evaluate
            print(f"     🔍 Evaluating Target...")
            target_res = evaluate_audio(target_path, methods)
            
            print(f"     🔍 Evaluating Trained...")
            trained_res = evaluate_audio(trained_path, methods)
            
            all_split_results[split_key] = {
                "target": target_res,
                "trained": trained_res,
                "target_file": os.path.basename(target_path),
                "trained_file": os.path.basename(trained_path)
            }

            # 2. Viz
            visualize_spectrograms(target_path, trained_path, output_dir, instrument_name, split_name=split_key)
            print(f"     ✅ Saved spectrogram comparison for {split_key}")
        else:
            if split_key == "train":
                print(f"   ⚠️ Missing train files: {target_path} or {trained_path}")

    # 3. Save JSON
    if all_split_results:
        evaluation_data = {
            "experiment": os.path.basename(exp_dir),
            "instrument": instrument_name,
            "splits": all_split_results
        }

        json_path = os.path.join(output_dir, "evaluation.json")
        with open(json_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        print(f"   ✅ Saved full evaluation to {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize Nasong training results.")
    parser.add_argument("--models-dir", type=str, default="trained_models", help="Directory containing experiment folders")
    parser.add_argument("--experiment", type=str, help="Specific experiment name to run")
    parser.add_argument("--target-wav", type=str, help="Specific target WAV file")
    parser.add_argument("--trained-wav", type=str, help="Specific trained WAV file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--methods", type=str, help="Comma-separated list of methods")

    args = parser.parse_args()

    methods = args.methods.split(",") if args.methods else ALL_METHODS

    # Mode 1: Specific Files
    if args.target_wav and args.trained_wav:
        out_dir = args.output_dir if args.output_dir else os.path.dirname(args.trained_wav)
        instrument_name = os.path.basename(args.target_wav).replace("_target.wav", "") or "unknown"

        print(f"📂 Processing specific files...")
        target_results = evaluate_audio(args.target_wav, methods)
        trained_results = evaluate_audio(args.trained_wav, methods)

        evaluation_data = {
            "experiment": "custom",
            "instrument": instrument_name,
            "target_audio": args.target_wav,
            "trained_audio": args.trained_wav,
            "results": {
                "target": target_results,
                "trained": trained_results
            }
        }

        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "evaluation.json")
        with open(json_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        print(f"   ✅ Saved evaluation to {json_path}")

        visualize_spectrograms(args.target_wav, args.trained_wav, out_dir, instrument_name)
        return

    # Mode 2: Iterate Experiments
    if args.experiment:
        experiments = [args.experiment]
    elif os.path.exists(args.models_dir):
        experiments = [d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]
    else:
        print(f"❌ Models directory not found: {args.models_dir}")
        return

    print(f"Found {len(experiments)} experiments in {args.models_dir}.")

    for exp in experiments:
        exp_dir = os.path.join(args.models_dir, exp)
        if not os.path.isdir(exp_dir):
             print(f"⚠️ {exp_dir} is not a directory. Skipping.")
             continue

        process_experiment(exp_dir, args.output_dir, methods)

if __name__ == "__main__":
    main()

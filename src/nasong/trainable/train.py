# Copyright (C) 2026 Nathan Cerisara <https://github.com/nath54/nasong>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Training script for Nasong synthesizer instruments.

This module provides a complete pipeline for training differentiable synthesizer
instruments to match a target audio file. It supports multiple training engines
(NumPy, Autograd, PyTorch) and automatic note detection.

Key Functions:
    get_engine:
        Factory that instantiates the requested training engine (numpy, autograd,
        or torch) from a ``TrainingConfig``.

    load_wav_segment:
        Loads a segment of a WAV file, normalises it to float32, resamples to a
        target sample rate, and pads/trims to the requested duration.

    render_audio_in_chunks:
        Renders audio from a ``Value`` synthesis graph in manageable chunks to
        avoid out-of-memory errors, using either PyTorch or NumPy indexing.

    train_instrument:
        End-to-end training loop. Loads audio, detects notes, builds the
        synthesis graph, runs the optimisation loop, and saves the resulting
        audio, parameters, and history to disk.

    main:
        CLI entry-point. Parses arguments (WAV file, YAML config, overrides)
        and launches ``train_instrument``.

Usage::

    # With a YAML config file
    python -m nasong.trainable.train --config my_config.yaml

    # Quick run with defaults
    python -m nasong.trainable.train target.wav -i sine -e 200 --engine autograd
"""

#
### Import Modules. ###
#
from typing import Any

#
import os
import argparse
import datetime
import scipy.io.wavfile as wavfile  # type: ignore

#
import numpy as np
from numpy.typing import NDArray

#
import nasong.core.all_values as lv
import nasong.trainable.extract as learnable
from nasong.trainable.config import TrainingConfig
from nasong.trainable.note_detection.create import create_note_detector

#
from nasong.trainable.engines.base import BaseTrainingEngine
from nasong.trainable.engines.numpy_engine import NumpyEngine
from nasong.trainable.engines.autograd_engine import AutogradEngine

#
### Try importing torch, handle failure gracefully ###
#
try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None  # type: ignore


#
def get_engine(config: TrainingConfig) -> BaseTrainingEngine:
    """Factory function to instantiate the requested training engine.

    Args:
        config (TrainingConfig): The global training configuration.

    Returns:
        BaseTrainingEngine: An instance of the selected engine (NumPy, Torch, etc.).

    Raises:
        ImportError: If the 'torch' engine is requested but PyTorch is not installed.
        ValueError: If the engine type is unrecognized.
    """

    if config.engine_type == "numpy":
        return NumpyEngine(config)
    elif config.engine_type == "autograd":
        return AutogradEngine(config)
    elif config.engine_type == "torch":
        # Lazy import to avoid crashing if torch is broken/missing
        from nasong.trainable.engines.torch_engine import TorchEngine

        if not HAS_TORCH:
            raise ImportError(
                "Cannot use TorchEngine because PyTorch is not available."
            )
        return TorchEngine(config)
    else:
        raise ValueError(f"Unknown engine type: {config.engine_type}")


#
### UTILITY FUNCTIONS ###
#


def load_wav_segment(
    wav_path: str,
    start_time: float = 0.0,
    duration: float = 5.0,
    target_sample_rate: int = 44100,
) -> tuple[NDArray[np.float32], int]:
    """Load a segment from a WAV file."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    sample_rate, audio_data = wavfile.read(wav_path)

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sample_rate != target_sample_rate:
        ratio = target_sample_rate / sample_rate
        new_length = int(len(audio_data) * ratio)
        audio_data = np.interp(
            np.linspace(0, len(audio_data) - 1, new_length),
            np.arange(len(audio_data)),
            audio_data,
        )
        sample_rate = target_sample_rate

    start_sample = int(start_time * sample_rate)
    duration_samples = int(duration * sample_rate)
    end_sample = min(start_sample + duration_samples, len(audio_data))

    segment = audio_data[start_sample:end_sample]

    if len(segment) < duration_samples:
        segment = np.pad(segment, (0, duration_samples - len(segment)), mode="constant")

    return segment.astype(np.float32), sample_rate


#
### TRAINING FUNCTION ###
#


def render_audio_in_chunks(
    synth_output: lv.Value,
    total_samples: int,
    sr: int,
    device: str,
    start_sample: int = 0,
    chunk_size_sec: float = 5.0,
) -> NDArray[np.float32]:
    """Render audio in chunks to avoid OOM."""
    chunk_size_samples = int(chunk_size_sec * sr)
    audio_chunks = []

    current = start_sample
    end_total = start_sample + total_samples

    while current < end_total:
        end = min(current + chunk_size_samples, end_total)

        if HAS_TORCH and (device != "cpu" or True):
            # indices relative to global time for the graph
            idx_torch = torch.arange(current, end, dtype=torch.float32, device=device)  # pylint: disable=no-member
            chunk = (
                synth_output.getitem_torch(idx_torch, sr, device=device)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            # Fallback to NumPy
            idx_np = np.arange(current, end, dtype=np.float32)
            chunk = synth_output.getitem_np(idx_np, sr)

        audio_chunks.append(chunk)
        current = end

    return np.concatenate(audio_chunks)


def train_instrument(config: TrainingConfig) -> dict[str, Any]:
    """
    Train an instrument using the configured Engine.
    """
    print(f"=== Training {config.instrument_name} on {config.target_wav} ===")
    print(f"=== Engine: {config.engine_type} ===")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Avoid double appending if already present (simple heuristic)
    # But since we want every run to be unique, we just append.
    config.output_dir = os.path.join(config.output_dir, timestamp)

    # Load target audio
    # Determine total duration
    total_duration = config.train_duration + config.val_duration + config.test_duration
    if total_duration <= 0:
        total_duration = config.audio.duration

    print(
        f"Loading total audio segment ({config.audio.start_time}s - {config.audio.start_time + total_duration}s)..."
    )
    full_audio, sr = load_wav_segment(
        config.target_wav,
        config.audio.start_time,
        total_duration,
        config.audio.sample_rate,
    )

    # Split Audio
    train_end_sample = int(config.train_duration * sr)
    val_end_sample = train_end_sample + int(config.val_duration * sr)

    train_audio = full_audio[:train_end_sample]
    val_audio = (
        full_audio[train_end_sample:val_end_sample]
        if config.val_duration > 0
        else np.array([])
    )
    test_audio = (
        full_audio[val_end_sample:] if config.test_duration > 0 else np.array([])
    )

    print(
        f"Split sizes: Train={len(train_audio) / sr:.2f}s, Val={len(val_audio) / sr:.2f}s, Test={len(test_audio) / sr:.2f}s"
    )

    # Run Detection
    print(f"Detecting notes using {config.note_detection.method} on full audio...")
    detector = create_note_detector(config.note_detection)
    all_notes = detector.detect(full_audio, sr)

    print(f"Detected {len(all_notes)} notes.")
    if not all_notes and config.note_detection.method == "legacy":
        # Fallback for legacy
        all_notes = [
            {
                "frequencies": [261.63],
                "start_time": 0.0,
                "duration": total_duration * 0.9,
            }
        ]

    # Filter notes for TRAIN split
    train_notes = []
    for note in all_notes:
        if note["start_time"] < config.train_duration:
            train_notes.append(note)

    print(f"Training notes: {len(train_notes)}")

    # Initialize Trainable Parameters
    trainable_notes = []
    for i, note in enumerate(train_notes):
        note_prefix = f"note_{i}"

        freq_params = []
        for j, f in enumerate(note["frequencies"]):
            freq_params.append(
                lv.ValueTrainableParameter(f, name=f"{note_prefix}_freq_{j}")
            )

        trainable_notes.append(
            {
                "frequencies": freq_params,
                "start_time": lv.ValueTrainableParameter(
                    note["start_time"], name=f"{note_prefix}_start"
                ),
                "duration": lv.ValueTrainableParameter(
                    note["duration"], name=f"{note_prefix}_dur"
                ),
                "amplitude": note.get("amplitude", 0.5),
                "prefix": note_prefix,
            }
        )

    # Build Instrument Graph (Training)
    print(f"Building {config.instrument_name} synthesis graph (Training)...")
    instrument_blueprint = learnable.get_trainable_instrument(config.instrument_name)

    time_val = lv.BasicScaling(
        value=lv.Identity(), mult_scale=lv.Constant(1 / sr), sum_scale=lv.Constant(0)
    )

    note_vals = []
    for tn in trainable_notes:
        prefix = tn["prefix"]

        if config.instrument_name in ["kick", "snare", "hihat_closed", "hihat_open"]:
            #
            if config.instrument_name == "hihat_open":
                nv = instrument_blueprint(
                    time=time_val,
                    start_time=float(tn["start_time"].value),
                    is_open=True,
                    name_prefix=prefix,
                )
            elif config.instrument_name == "hihat_closed":
                nv = instrument_blueprint(
                    time=time_val,
                    start_time=float(tn["start_time"].value),
                    is_open=False,
                    name_prefix=prefix,
                )
            else:
                nv = instrument_blueprint(
                    time=time_val,
                    start_time=float(tn["start_time"].value),
                    name_prefix=prefix,
                )
            note_vals.append(nv)
        else:
            chord_voices = []
            for j, fp in enumerate(tn["frequencies"]):
                voice_prefix = f"{prefix}_v{j}"
                voice = instrument_blueprint(
                    time=time_val,
                    frequency=fp,
                    start_time=float(tn["start_time"].value),
                    duration=float(tn["duration"].value),
                    init_amplitude=float(tn["amplitude"]),
                    name_prefix=voice_prefix,
                )
                chord_voices.append(voice)
            if len(chord_voices) == 1:
                note_vals.append(chord_voices[0])
            else:
                note_vals.append(
                    lv.Product(
                        lv.Sum(chord_voices), lv.Constant(1.0 / len(chord_voices))
                    )
                )

    if not note_vals:
        print("Error: No training notes.")
        return {}

    synth_output = lv.Sum(note_vals) if len(note_vals) > 1 else note_vals[0]

    # --- ENGINE INITIALIZATION ---
    engine = get_engine(config)

    # Initialize implementation-specific details if needed (e.g. Torch optimizer)
    if hasattr(engine, "initialize_optimizer"):
        engine.initialize_optimizer(synth_output)

    history: dict[str, list[float]] = {
        "losses": [],
        "epochs": [],
        "validation_losses": [],
    }

    print(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        #
        loss_val = engine.compute_loss(train_audio, synth_output, sr)
        metrics = engine.step()

        avg_loss = metrics.get("loss", loss_val)

        history["losses"].append(avg_loss)
        history["epochs"].append(epoch)

        if epoch % 1 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs} | Loss: {avg_loss:.6f}")

    print("\n=== Training Complete ===")

    # Save Artifacts
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Config
    if config.save_config:
        config.to_yaml(os.path.join(config.output_dir, "config.yaml"))

    # 2. History
    if config.save_history:
        import json

        with open(
            os.path.join(config.output_dir, "history.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)

    # 3. Save Audio (All Splits)
    print("Rendering audio for all splits...")

    def save_split_audio(audio_data, suffix, target_audio=None):
        if len(audio_data) == 0:
            return

        path = os.path.join(config.output_dir, f"{config.instrument_name}_{suffix}.wav")
        wavfile.write(path, sr, (audio_data * 32767).astype(np.int16))

        if target_audio is not None and len(target_audio) > 0:
            target_path = os.path.join(
                config.output_dir, f"{config.instrument_name}_{suffix}_target.wav"
            )
            wavfile.write(target_path, sr, (target_audio * 32767).astype(np.int16))

    # Render TRAIN
    train_pred = render_audio_in_chunks(
        synth_output, len(train_audio), sr, config.device, start_sample=0
    )
    save_split_audio(train_pred, "trained", train_audio)

    # Render VAL
    if len(val_audio) > 0:
        val_pred = render_audio_in_chunks(
            synth_output,
            len(val_audio),
            sr,
            config.device,
            start_sample=train_end_sample,
        )
        save_split_audio(val_pred, "val_trained", val_audio)

    # Render TEST
    if len(test_audio) > 0:
        test_pred = render_audio_in_chunks(
            synth_output,
            len(test_audio),
            sr,
            config.device,
            start_sample=val_end_sample,
        )
        save_split_audio(test_pred, "test_trained", test_audio)

    print(f"Saved audio artifacts to: {config.output_dir}")

    # 4. Save Params
    # Use Engine's method
    param_dict = engine.get_parameter_values()

    # Save as generic params.json for inference compatibility
    with open(
        os.path.join(config.output_dir, "params.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(param_dict, f, indent=2)

    # Also save with instrument name for reference (optional, keeping backward compat if needed)
    with open(
        os.path.join(config.output_dir, f"{config.instrument_name}_params.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(param_dict, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train Nasong instruments using Modular Engines."
    )

    parser.add_argument("wav_file", nargs="?", help="Path to input WAV file")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")

    # Overrides
    parser.add_argument("--instrument", "-i", type=str, help="Instrument name")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="Note detection method (legacy, basic_pitch_onnx, librosa, torchcrepe)",
    )
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda)")
    parser.add_argument(
        "--engine", type=str, help="Engine type (autograd, numpy, torch)"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}...")
        config = TrainingConfig.from_yaml(args.config)

        # Apply overrides
        if args.instrument:
            config.instrument_name = args.instrument
        if args.wav_file:
            config.target_wav = args.wav_file
        if args.epochs:
            config.epochs = args.epochs
        if args.lr:
            config.learning_rate = args.lr
        if args.method:
            config.note_detection.method = args.method
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.device:
            config.device = args.device
        if args.engine:
            config.engine_type = args.engine

        if config.engine_type == "torch" and not HAS_TORCH:
            print("Torch is not available, switching to autograd.")
            config.engine_type = "autograd"

        if config.engine_type == "torch":
            # Fallback to CPU if CUDA is not available
            if config.device == "cuda" and not torch.cuda.is_available():  # pylint: disable=no-member
                config.device = "cpu"
                print("CUDA not available. Switching to CPU.")

    else:
        if not args.wav_file:
            parser.print_help()
            print("\nError: WAV file or config file required.")
            return

        print("No config file provided. Using defaults with overrides.")
        config = TrainingConfig(
            instrument_name=args.instrument if args.instrument else "sine",
            target_wav=args.wav_file,
            epochs=args.epochs if args.epochs else 100,
            learning_rate=args.lr if args.lr else 0.01,
            output_dir=args.output_dir if args.output_dir else "trained_models",
            device=args.device if args.device else "cpu",
            engine_type=args.engine if args.engine else "autograd",
        )

        if args.method:
            config.note_detection.method = args.method

        if config.engine_type == "torch" and not HAS_TORCH:
            print("Torch is not available, switching to autograd.")
            config.engine_type = "autograd"

        if config.engine_type == "torch":
            # Fallback to CPU if CUDA is not available
            if config.device == "cuda" and not torch.cuda.is_available():  # pylint: disable=no-member
                config.device = "cpu"
                print("CUDA not available. Switching to CPU.")

    # Validate
    if not os.path.exists(config.target_wav):
        print(f"Error: Target WAV file not found: {config.target_wav}")
        return

    # Run
    train_instrument(config)


if __name__ == "__main__":
    main()

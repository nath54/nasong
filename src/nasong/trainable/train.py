import os
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.optim as optim
import scipy.io.wavfile as wavfile

import nasong.core.value as lv
import nasong.trainable.extract as learnable
from nasong.trainable.config import TrainingConfig, NoteDetectionConfig
from nasong.trainable.note_detection.create import create_note_detector

#
### UTILITY FUNCTIONS ###
#

def load_wav_segment(
    wav_path: str,
    start_time: float = 0.0,
    duration: float = 5.0,
    target_sample_rate: int = 44100,
) -> Tuple[NDArray[np.float32], int]:
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

def spectral_loss(
    synthesized: Tensor,
    target: Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    high_freq_emphasis: float = 2.0,
) -> Tensor:
    synth_stft = torch.stft(
        synthesized,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=synthesized.device),
        return_complex=True,
    )
    target_stft = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=target.device),
        return_complex=True,
    )

    synth_mag = torch.abs(synth_stft)
    target_mag = torch.abs(target_stft)

    freq_bins = synth_mag.shape[0]
    freq_weights = torch.linspace(
        1.0, high_freq_emphasis, freq_bins, device=synthesized.device
    )
    freq_weights = freq_weights.unsqueeze(1)

    synth_mag_weighted = synth_mag * freq_weights
    target_mag_weighted = target_mag * freq_weights

    mag_loss = torch.mean(torch.abs(synth_mag_weighted - target_mag_weighted))

    synth_log_mag = torch.log(synth_mag + 1e-5)
    target_log_mag = torch.log(target_mag + 1e-5)
    log_mag_loss = torch.mean(torch.abs(synth_log_mag - target_log_mag))

    total_loss = mag_loss + 0.5 * log_mag_loss

    return total_loss

def multi_resolution_spectral_loss(
    synthesized: Tensor,
    target: Tensor,
    sample_rate: int = 44100,
    fft_sizes: List[int] = None,
    high_freq_emphasis: float = 2.0,
) -> Tensor:
    if fft_sizes is None:
        fft_sizes = [2048, 1024, 512]

    total_loss = 0.0

    for n_fft in fft_sizes:
        hop_length = n_fft // 4
        loss = spectral_loss(
            synthesized, target, sample_rate, n_fft, hop_length, high_freq_emphasis
        )
        total_loss = total_loss + loss

    return total_loss / len(fft_sizes)

def collect_trainable_parameters(
    value: lv.Value, params: List[Tensor] = None
) -> List[Tensor]:
    if params is None:
        params = []

    if isinstance(value, lv.ValueTrainableParameter):
        if value.value not in params:
            params.append(value.value)

    for attr_name in dir(value):
        if attr_name.startswith("_"):
            continue

        try:
            attr = getattr(value, attr_name)

            if isinstance(attr, lv.Value):
                collect_trainable_parameters(attr, params)

            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, lv.Value):
                        collect_trainable_parameters(item, params)
                    elif isinstance(item, tuple) and len(item) == 2:
                        for sub_item in item:
                            if isinstance(sub_item, lv.Value):
                                collect_trainable_parameters(sub_item, params)
        except Exception:
            continue

    return params


#
### TRAINING FUNCTION ###
#

def train_instrument(config: TrainingConfig) -> Dict[str, Any]:
    """
    Train an instrument to match a WAV file segment using the provided configuration.
    """
    print(f"=== Training {config.instrument_name} on {config.target_wav} ===")

    # Load target audio
    print(f"Loading audio segment ({config.audio.start_time}s - {config.audio.start_time + config.audio.duration}s)...")
    target_audio, sr = load_wav_segment(
        config.target_wav,
        config.audio.start_time,
        config.audio.duration,
        config.audio.sample_rate
    )

    target_tensor = torch.from_numpy(target_audio).to(
        dtype=torch.float32, device=config.device
    )

    # Extract note parameters using configured detector
    print(f"Detecting notes using {config.note_detection.method}...")

    detector = create_note_detector(config.note_detection)

    # Note: detectors might need full audio or handle segmentation internally.
    # Current LegacyDetector expects the segment.
    # Librosa can handle segment. BasicPitch works best on segments too if short.
    note_params = detector.detect(target_audio, sr)

    print("Detected notes:")
    if not note_params:
        print("  No notes detected! Using default C4.")
        note_params = [{
            "frequencies": [261.63],
            "start_time": 0.0,
            "duration": config.audio.duration * 0.9
        }]

    for i, note in enumerate(note_params):
        freq_str = ", ".join([f"{f:.1f}Hz" for f in note["frequencies"]])
        print(
            f"  Note {i + 1}: [{freq_str}], start={note['start_time']:.2f}s, dur={note['duration']:.2f}s"
        )

    # Create trainable note parameters
    print("\nInitializing trainable note parameters...")
    trainable_notes = []

    # Optional: Filter detected notes to fit segment if they exceed bounds?
    # For now assume detector returns relative to the segment start or correct times.
    # Note: Detectors return time relative to the START of the provided audio segment.
    # The audio segment already starts at 0.0 relative to itself.

    for note in note_params:
        freq_list = note["frequencies"]
        # Basic Pitch might return a chord as separate notes with same start time.
        # Legacy returns groups.
        # We handle separate notes naturally by adding multiple entries.

        freq_params = [lv.ValueTrainableParameter(f) for f in freq_list]
        trainable_notes.append(
            {
                "frequencies": freq_params,
                "start_time": lv.ValueTrainableParameter(note["start_time"]),
                "duration": lv.ValueTrainableParameter(note["duration"]),
            }
        )

    # Build instrument synthesis graph
    print(f"Building {config.instrument_name} synthesis graph...")
    instrument_blueprint = learnable.get_trainable_instrument(config.instrument_name)

    time_val = lv.BasicScaling(
        value=lv.Identity(),
        mult_scale=lv.Constant(1 / sr),
        sum_scale=lv.Constant(0),
    )

    note_values = []
    for tn in trainable_notes:
        if config.instrument_name in ["kick", "snare", "hihat_closed", "hihat_open"]:
            note_val = instrument_blueprint(
                time=time_val, start_time=tn["start_time"].value.item()
            )
            note_values.append(note_val)
        else:
            chord_voices = []
            for freq_param in tn["frequencies"]:
                voice = instrument_blueprint(
                    time=time_val,
                    frequency=freq_param,
                    start_time=tn["start_time"].value.item(),
                    duration=tn["duration"].value.item(),
                )
                chord_voices.append(voice)

            if len(chord_voices) == 1:
                note_values.append(chord_voices[0])
            else:
                note_values.append(
                    lv.Product(
                        lv.Sum(chord_voices), lv.Constant(1.0 / len(chord_voices))
                    )
                )

    if not note_values:
        print("Error: No notes to synthesize.")
        return {}

    if len(note_values) == 1:
        synth_output = note_values[0]
    else:
        synth_output = lv.Sum(note_values)

    print("Collecting trainable parameters...")
    all_params = collect_trainable_parameters(synth_output)

    for tn in trainable_notes:
        for freq_param in tn["frequencies"]:
            all_params.append(freq_param.value)
        all_params.append(tn["start_time"].value)
        all_params.append(tn["duration"].value)

    all_params = list(set(all_params))
    for param in all_params:
        param.requires_grad = True

    print(f"Total trainable parameters: {len(all_params)}")

    optimizer = optim.Adam(all_params, lr=config.learning_rate)

    print(f"\nStarting training for {config.epochs} epochs...")
    history = {"losses": [], "epochs": []}

    for epoch in range(config.epochs):
        optimizer.zero_grad()

        idx_buffer = torch.arange(
            len(target_tensor), dtype=torch.float32, device=config.device
        )
        synthesized = synth_output.getitem_torch(idx_buffer, sr, device=config.device)

        loss = multi_resolution_spectral_loss(
            synthesized,
            target_tensor,
            sample_rate=sr,
            fft_sizes=config.spectral_loss.fft_sizes,
            high_freq_emphasis=config.spectral_loss.high_freq_emphasis
        )

        loss.backward()
        optimizer.step()

        history["losses"].append(loss.item())
        history["epochs"].append(epoch)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs} | Spectral Loss: {loss.item():.6f}")

    print("\n=== Training Complete ===")

    os.makedirs(config.output_dir, exist_ok=True)

    final_idx = torch.arange(len(target_tensor), dtype=torch.float32, device=config.device)
    final_audio = (
        synth_output.getitem_torch(final_idx, sr, device=config.device)
        .detach()
        .cpu()
        .numpy()
    )

    output_path = os.path.join(config.output_dir, f"{config.instrument_name}_trained.wav")
    wavfile.write(output_path, sr, (final_audio * 32767).astype(np.int16))
    print(f"Saved synthesized audio to: {output_path}")

    target_path = os.path.join(config.output_dir, f"{config.instrument_name}_target.wav")
    wavfile.write(target_path, sr, (target_audio * 32767).astype(np.int16))
    print(f"Saved target audio to: {target_path}")

    param_dict = {}
    for i, param in enumerate(all_params):
        param_dict[f"param_{i}"] = param.detach().cpu().item()

    import json
    param_path = os.path.join(config.output_dir, f"{config.instrument_name}_params.json")
    with open(param_path, "w") as f:
        json.dump(param_dict, f, indent=2)
    print(f"Saved parameters to: {param_path}")

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train Nasong instruments with configurable note detection."
    )

    parser.add_argument("wav_file", nargs='?', help="Path to input WAV file")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")

    # Overrides
    parser.add_argument("--instrument", "-i", type=str, help="Instrument name")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--method", "-m", type=str, help="Note detection method (legacy, basic_pitch_onnx, librosa, torchcrepe)")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda)")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}...")
        config = TrainingConfig.from_yaml(args.config)

        # Apply overrides
        if args.instrument: config.instrument_name = args.instrument
        if args.wav_file: config.target_wav = args.wav_file
        if args.epochs: config.epochs = args.epochs
        if args.lr: config.learning_rate = args.lr
        if args.method: config.note_detection.method = args.method
        if args.output_dir: config.output_dir = args.output_dir
        if args.device: config.device = args.device

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
            device=args.device if args.device else "cpu"
        )
        if args.method:
            config.note_detection.method = args.method

    # Validate
    if not os.path.exists(config.target_wav):
         print(f"Error: Target WAV file not found: {config.target_wav}")
         return

    # Run
    train_instrument(config)


if __name__ == "__main__":
    main()

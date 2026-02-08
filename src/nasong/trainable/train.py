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

def render_audio_in_chunks(
    synth_output: lv.Value,
    total_samples: int,
    sr: int,
    device: str,
    start_sample: int = 0,
    chunk_size_sec: float = 5.0
) -> NDArray[np.float32]:
    """Render audio in chunks to avoid OOM."""
    chunk_size_samples = int(chunk_size_sec * sr)
    audio_chunks = []
    
    current = start_sample
    end_total = start_sample + total_samples
    
    while current < end_total:
        end = min(current + chunk_size_samples, end_total)
        # indices relative to global time for the graph
        idx = torch.arange(current, end, dtype=torch.float32, device=device)
        
        chunk = synth_output.getitem_torch(idx, sr, device=device).detach().cpu().numpy()
        audio_chunks.append(chunk)
        current = end
        
    return np.concatenate(audio_chunks)


def train_instrument(config: TrainingConfig) -> Dict[str, Any]:
    """
    Train an instrument to match a WAV file segment using the provided configuration.
    """
    print(f"=== Training {config.instrument_name} on {config.target_wav} ===")

    # Load target audio
    # Determine total duration
    total_duration = config.train_duration + config.val_duration + config.test_duration
    if total_duration <= 0:
        total_duration = config.audio.duration

    print(f"Loading total audio segment ({config.audio.start_time}s - {config.audio.start_time + total_duration}s)...")
    full_audio, sr = load_wav_segment(
        config.target_wav,
        config.audio.start_time,
        total_duration,
        config.audio.sample_rate
    )

    # Split Audio
    train_end_sample = int(config.train_duration * sr)
    val_end_sample = train_end_sample + int(config.val_duration * sr)

    train_audio = full_audio[:train_end_sample]
    val_audio = full_audio[train_end_sample:val_end_sample] if config.val_duration > 0 else np.array([])
    test_audio = full_audio[val_end_sample:] if config.test_duration > 0 else np.array([])

    print(f"Split sizes: Train={len(train_audio)/sr:.2f}s, Val={len(val_audio)/sr:.2f}s, Test={len(test_audio)/sr:.2f}s")

    # Run Detection on FULL audio (better context)
    print(f"Detecting notes using {config.note_detection.method} on full audio...")
    detector = create_note_detector(config.note_detection)
    all_notes = detector.detect(full_audio, sr)

    print(f"Detected {len(all_notes)} notes.")
    if not all_notes and config.note_detection.method == 'legacy':
         # Fallback for legacy
         all_notes = [{
            "frequencies": [261.63],
            "start_time": 0.0,
            "duration": total_duration * 0.9
        }]

    # Filter notes for TRAIN split
    train_notes = []
    for note in all_notes:
        # Keep notes that start within train window
        if note['start_time'] < config.train_duration:
            # Clip duration if it extends beyond
            # Actually, standard is allow decay. But here we just classify.
            train_notes.append(note)

    print(f"Training notes: {len(train_notes)}")

    # Initialize Trainable Parameters
    # Convert filtered notes to ValueTrainableParameter
    trainable_notes = []
    for note in train_notes:
         freq_params = [lv.ValueTrainableParameter(f) for f in note["frequencies"]]
         trainable_notes.append({
             "frequencies": freq_params,
             "start_time": lv.ValueTrainableParameter(note["start_time"]),
             "duration": lv.ValueTrainableParameter(note["duration"]),
         })

    # Build Instrument Graph (Training)
    print(f"Building {config.instrument_name} synthesis graph (Training)...")
    instrument_blueprint = learnable.get_trainable_instrument(config.instrument_name)

    # We need a synthesis graph that can handle the full duration or be batched.
    # If we batch, we need to slice time.
    # Standard Nasong `time` goes 0..T.
    # The note `start_time` is relative to 0.

    # Building ONE big graph for training set to manage parameters
    # Memory optimization: The graph structure itself is cheap. Rendering big buffers is expensive.

    time_val = lv.BasicScaling(value=lv.Identity(), mult_scale=lv.Constant(1/sr), sum_scale=lv.Constant(0))

    note_vals = []
    for tn in trainable_notes:
        if config.instrument_name in ["kick", "snare", "hihat_closed", "hihat_open"]:
            nv = instrument_blueprint(time=time_val, start_time=tn["start_time"].value.item())
            note_vals.append(nv)
        else:
            chord_voices = []
            for fp in tn["frequencies"]:
                 # IMPORTANT: To share parameters, we pass the parameter object!
                 voice = instrument_blueprint(
                     time=time_val, frequency=fp,
                     start_time=tn["start_time"].value.item(),
                     duration=tn["duration"].value.item()
                 )
                 chord_voices.append(voice)
            if len(chord_voices) == 1: note_vals.append(chord_voices[0])
            else: note_vals.append(lv.Product(lv.Sum(chord_voices), lv.Constant(1.0/len(chord_voices))))

    if not note_vals:
        print("Error: No training notes.")
        return {}

    synth_output = lv.Sum(note_vals) if len(note_vals) > 1 else note_vals[0]

    # Collect Params
    print("Collecting parameters...")
    all_params = collect_trainable_parameters(synth_output)
    # Add note params explicitly
    for tn in trainable_notes:
        for fp in tn["frequencies"]: all_params.append(fp.value)
        all_params.append(tn["start_time"].value)
        all_params.append(tn["duration"].value)

    all_params = list(set(all_params))
    for p in all_params: p.requires_grad = True
    print(f"Total trainable parameters: {len(all_params)}")

    optimizer = optim.Adam(all_params, lr=config.learning_rate)
    history = {"losses": [], "epochs": [], "validation_losses": []}

    # BATCH SETUP
    batch_size_samples = int(config.batch_duration * sr)
    overlap_samples = int(config.batch_overlap * sr)
    stride = batch_size_samples - overlap_samples
    if stride <= 0: stride = batch_size_samples // 2

    train_tensor = torch.from_numpy(train_audio).to(device=config.device)

    print(f"Starting training for {config.epochs} epochs (Batch: {config.batch_duration}s, Overlap: {config.batch_overlap}s)...")

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        batches = 0

        # Iterate windows
        current_sample = 0
        while current_sample < len(train_tensor):
            end_sample = min(current_sample + batch_size_samples, len(train_tensor))
            if end_sample - current_sample < 1024: break # Skip tiny last batch

            # Slice Target
            target_batch = train_tensor[current_sample:end_sample]

            # Index buffer for synthesis (absolute time)
            # nasong uses float index? No, getitem_torch uses indices.
            # We enable the graph to render specifically for these indices.
            idx_buffer = torch.arange(current_sample, end_sample, dtype=torch.float32, device=config.device)

            # Render
            # This efficiently only computes for this time window!
            synthesized = synth_output.getitem_torch(idx_buffer, sr, device=config.device)

            loss = multi_resolution_spectral_loss(
                synthesized, target_batch, sr,
                config.spectral_loss.fft_sizes,
                config.spectral_loss.high_freq_emphasis
            )

            loss.backward()
            epoch_loss += loss.item()
            batches += 1

            current_sample += stride

        optimizer.step()

        avg_loss = epoch_loss / max(1, batches)
        history["losses"].append(avg_loss)
        history["epochs"].append(epoch)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}/{config.epochs} | Avg Loss: {avg_loss:.6f}")

    print("\n=== Training Complete ===")

    # Save Artifacts
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Config
    if config.save_config:
        config.to_yaml(os.path.join(config.output_dir, "config.yaml"))

    # 2. History
    if config.save_history:
        import json
        with open(os.path.join(config.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # 3. Save Audio (All Splits)
    print("Rendering audio for all splits...")
    
    # helper to save split
    def save_split_audio(audio_data, suffix, target_audio=None):
        if len(audio_data) == 0: return
        
        path = os.path.join(config.output_dir, f"{config.instrument_name}_{suffix}.wav")
        wavfile.write(path, sr, (audio_data * 32767).astype(np.int16))
        
        if target_audio is not None and len(target_audio) > 0:
            target_path = os.path.join(config.output_dir, f"{config.instrument_name}_{suffix}_target.wav")
            wavfile.write(target_path, sr, (target_audio * 32767).astype(np.int16))

    # Render TRAIN
    train_pred = render_audio_in_chunks(synth_output, len(train_audio), sr, config.device, start_sample=0)
    save_split_audio(train_pred, "trained", train_audio)

    # Render VAL
    if len(val_audio) > 0:
        val_pred = render_audio_in_chunks(synth_output, len(val_audio), sr, config.device, start_sample=train_end_sample)
        save_split_audio(val_pred, "val_trained", val_audio)

    # Render TEST
    if len(test_audio) > 0:
        test_pred = render_audio_in_chunks(synth_output, len(test_audio), sr, config.device, start_sample=val_end_sample)
        save_split_audio(test_pred, "test_trained", test_audio)

    print(f"Saved audio artifacts to: {config.output_dir}")
    # 4. Save Params
    param_dict = {}
    for i, param in enumerate(all_params):
        param_dict[f"param_{i}"] = param.detach().cpu().item()
    with open(os.path.join(config.output_dir, f"{config.instrument_name}_params.json"), "w") as f:
        json.dump(param_dict, f, indent=2)

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

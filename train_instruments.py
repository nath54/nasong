#
### Import Modules. ###
#
import os
import argparse
from typing import List, Tuple, Dict, Any

#
import numpy as np
from numpy.typing import NDArray

#
import torch
from torch import Tensor
import torch.optim as optim

#
import scipy.io.wavfile as wavfile

#
import lib_value as lv
import lib_extr_instr_to_learn as learnable
import lib_song as ls


#
### UTILITY FUNCTIONS ###
#


def load_wav_segment(
    wav_path: str,
    start_time: float = 0.0,
    duration: float = 5.0,
    target_sample_rate: int = 44100,
) -> Tuple[NDArray[np.float32], int]:
    """
    Load a segment from a WAV file.

    Args:
        wav_path: Path to WAV file
        start_time: Start time in seconds
        duration: Duration in seconds
        target_sample_rate: Target sample rate

    Returns:
        (audio_data, sample_rate) tuple
    """

    # Load WAV file
    sample_rate, audio_data = wavfile.read(wav_path)

    # Convert to float32 and normalize
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if needed (simple decimation/interpolation)
    if sample_rate != target_sample_rate:
        # Simple resampling - for production use scipy.signal.resample
        ratio = target_sample_rate / sample_rate
        new_length = int(len(audio_data) * ratio)
        audio_data = np.interp(
            np.linspace(0, len(audio_data) - 1, new_length),
            np.arange(len(audio_data)),
            audio_data,
        )
        sample_rate = target_sample_rate

    # Extract segment
    start_sample = int(start_time * sample_rate)
    duration_samples = int(duration * sample_rate)
    end_sample = min(start_sample + duration_samples, len(audio_data))

    segment = audio_data[start_sample:end_sample]

    # Pad if too short
    if len(segment) < duration_samples:
        segment = np.pad(segment, (0, duration_samples - len(segment)), mode="constant")

    return segment.astype(np.float32), sample_rate


def extract_note_parameters(
    audio_segment: NDArray[np.float32],
    sample_rate: int,
    num_notes: int = 1,
    use_onset_detection: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract note parameters using onset detection and pitch tracking.

    Args:
        audio_segment: Audio data
        sample_rate: Sample rate
        num_notes: Expected number of notes
        use_onset_detection: Use onset detection (True) or simple division (False)

    Returns:
        List of dicts with 'frequencies' (list), 'start_time', 'duration'
    """

    total_duration = len(audio_segment) / sample_rate
    notes = []

    if use_onset_detection:
        # Simple energy-based onset detection
        # Calculate short-time energy
        frame_length = int(0.02 * sample_rate)  # 20ms frames
        hop_length = int(0.01 * sample_rate)  # 10ms hop

        energy = []
        for i in range(0, len(audio_segment) - frame_length, hop_length):
            frame = audio_segment[i : i + frame_length]
            energy.append(np.sum(frame**2))

        energy = np.array(energy)

        # Find peaks in energy (onsets)
        # Simple threshold-based detection
        threshold = 0.3 * np.max(energy)
        onsets = []
        for i in range(1, len(energy) - 1):
            if (
                energy[i] > threshold
                and energy[i] > energy[i - 1]
                and energy[i] > energy[i + 1]
            ):
                onset_time = i * hop_length / sample_rate
                onsets.append(onset_time)

        # Limit to num_notes
        onsets = onsets[:num_notes]

        # If no onsets detected, fall back to simple division
        if len(onsets) == 0:
            onsets = [i * total_duration / num_notes for i in range(num_notes)]
    else:
        # Simple even division
        onsets = [i * total_duration / num_notes for i in range(num_notes)]

    # Extract pitch and duration for each onset
    for i, start_time in enumerate(onsets):
        # Determine duration (to next onset or end)
        if i < len(onsets) - 1:
            duration = onsets[i + 1] - start_time
        else:
            duration = total_duration - start_time

        # Extract segment for pitch detection
        start_idx = int(start_time * sample_rate)
        # Use first 100ms or duration, whichever is shorter
        analysis_duration = min(0.1, duration)
        end_idx = int((start_time + analysis_duration) * sample_rate)
        note_segment = audio_segment[start_idx:end_idx]

        # Detect multiple pitches (chord detection)
        frequencies = detect_pitches_fft(note_segment, sample_rate, max_pitches=3)

        notes.append(
            {
                "frequencies": frequencies,  # List of frequencies (chord)
                "start_time": start_time,
                "duration": duration * 0.9,  # Leave small gap
            }
        )

    return notes


def detect_pitches_fft(
    audio_segment: NDArray[np.float32],
    sample_rate: int,
    max_pitches: int = 3,
    min_freq: float = 50.0,
    max_freq: float = 4000.0,
) -> List[float]:
    """
    Detect multiple pitches using FFT peak detection.

    Args:
        audio_segment: Audio segment to analyze
        sample_rate: Sample rate
        max_pitches: Maximum number of pitches to detect
        min_freq: Minimum frequency to consider
        max_freq: Maximum frequency to consider

    Returns:
        List of detected frequencies (sorted by amplitude)
    """

    if len(audio_segment) == 0:
        return [220.0]  # Default A3

    # Apply window
    windowed = audio_segment * np.hanning(len(audio_segment))

    # FFT
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), 1 / sample_rate)
    magnitudes = np.abs(fft)

    # Focus on frequency range
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_filtered = freqs[freq_mask]
    mags_filtered = magnitudes[freq_mask]

    if len(mags_filtered) == 0:
        return [220.0]

    # Find peaks
    detected_freqs = []
    threshold = 0.1 * np.max(mags_filtered)

    # Find local maxima
    for i in range(1, len(mags_filtered) - 1):
        if (
            mags_filtered[i] > threshold
            and mags_filtered[i] > mags_filtered[i - 1]
            and mags_filtered[i] > mags_filtered[i + 1]
        ):
            detected_freqs.append((freqs_filtered[i], mags_filtered[i]))

    # Sort by magnitude and take top max_pitches
    detected_freqs.sort(key=lambda x: x[1], reverse=True)
    detected_freqs = detected_freqs[:max_pitches]

    # Extract just frequencies, sorted by frequency
    result = [f for f, _ in detected_freqs]
    result.sort()

    if len(result) == 0:
        return [220.0]

    return result


def spectral_loss(
    synthesized: Tensor,
    target: Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    high_freq_emphasis: float = 2.0,
) -> Tensor:
    """
    Multi-resolution spectral loss with high-frequency emphasis.

    Computes loss in frequency domain which is much better for audio than MSE.
    Emphasizes high frequencies more because they're perceptually important.

    Args:
        synthesized: Synthesized audio tensor
        target: Target audio tensor
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        high_freq_emphasis: Factor to emphasize high frequencies (>1 means more emphasis)

    Returns:
        Combined spectral loss
    """

    # Compute STFT for both signals
    # Note: torch.stft requires specific format
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

    # Get magnitudes
    synth_mag = torch.abs(synth_stft)
    target_mag = torch.abs(target_stft)

    # Frequency weighting (emphasize high frequencies)
    # Create weights that increase with frequency
    freq_bins = synth_mag.shape[0]
    freq_weights = torch.linspace(
        1.0, high_freq_emphasis, freq_bins, device=synthesized.device
    )
    freq_weights = freq_weights.unsqueeze(1)  # Shape: (freq_bins, 1)

    # Apply frequency weighting
    synth_mag_weighted = synth_mag * freq_weights
    target_mag_weighted = target_mag * freq_weights

    # Magnitude loss (L1)
    mag_loss = torch.mean(torch.abs(synth_mag_weighted - target_mag_weighted))

    # Log magnitude loss (better for perceptual quality)
    synth_log_mag = torch.log(synth_mag + 1e-5)
    target_log_mag = torch.log(target_mag + 1e-5)
    log_mag_loss = torch.mean(torch.abs(synth_log_mag - target_log_mag))

    # Combine losses
    total_loss = mag_loss + 0.5 * log_mag_loss

    return total_loss


def multi_resolution_spectral_loss(
    synthesized: Tensor,
    target: Tensor,
    sample_rate: int = 44100,
    fft_sizes: List[int] = None,
    high_freq_emphasis: float = 2.0,
) -> Tensor:
    """
    Multi-resolution spectral loss using multiple FFT sizes.

    This captures both fine and coarse spectral details.

    Args:
        synthesized: Synthesized audio
        target: Target audio
        sample_rate: Sample rate
        fft_sizes: List of FFT sizes to use
        high_freq_emphasis: High frequency emphasis factor

    Returns:
        Combined multi-resolution loss
    """

    if fft_sizes is None:
        fft_sizes = [2048, 1024, 512]  # Multiple resolutions

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
    """
    Recursively collect all ValueTrainableParameter tensors from a Value tree.

    Args:
        value: Root Value object
        params: Accumulator list

    Returns:
        List of all Tensor parameters
    """

    if params is None:
        params = []

    # Check if this is a trainable parameter
    if isinstance(value, lv.ValueTrainableParameter):
        if value.value not in params:  # Avoid duplicates
            params.append(value.value)

    # Recursively check all attributes that might be Values
    for attr_name in dir(value):
        if attr_name.startswith("_"):
            continue

        try:
            attr = getattr(value, attr_name)

            # Single Value
            if isinstance(attr, lv.Value):
                collect_trainable_parameters(attr, params)

            # List of Values
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, lv.Value):
                        collect_trainable_parameters(item, params)
                    elif isinstance(item, tuple) and len(item) == 2:
                        # Handle (weight, value) tuples in PonderedSum
                        for sub_item in item:
                            if isinstance(sub_item, lv.Value):
                                collect_trainable_parameters(sub_item, params)
        except:
            continue

    return params


#
### TRAINING FUNCTION ###
#


def train_instrument(
    wav_path: str,
    instrument_name: str,
    num_notes: int = 1,
    segment_start: float = 0.0,
    segment_duration: float = 5.0,
    sample_rate: int = 44100,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    output_dir: str = "trained_models",
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Train an instrument to match a WAV file segment.

    Args:
        wav_path: Path to target WAV file
        instrument_name: Name from TRAINABLE_INSTRUMENTS
        num_notes: Number of notes to extract
        segment_start: Start time of segment in WAV
        segment_duration: Duration of segment
        sample_rate: Sample rate for synthesis
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save outputs

    Returns:
        Dict with training history
    """

    print(f"=== Training {instrument_name} on {wav_path} ===")

    # Load target audio
    print(
        f"Loading audio segment ({segment_start}s - {segment_start + segment_duration}s)..."
    )
    target_audio, sr = load_wav_segment(
        wav_path, segment_start, segment_duration, sample_rate
    )
    target_tensor = torch.from_numpy(target_audio).to(
        dtype=torch.float32, device=device
    )

    # Extract note parameters (frequency, timing)
    print(f"Extracting {num_notes} note(s) from audio...")
    note_params = extract_note_parameters(
        target_audio, sample_rate, num_notes, use_onset_detection=True
    )

    print("Detected notes:")
    for i, note in enumerate(note_params):
        freq_str = ", ".join([f"{f:.1f}Hz" for f in note["frequencies"]])
        print(
            f"  Note {i + 1}: [{freq_str}], start={note['start_time']:.2f}s, dur={note['duration']:.2f}s"
        )

    # Create trainable note parameters (frequencies as list, start, duration)
    print("\nInitializing trainable note parameters...")
    trainable_notes = []
    for note in note_params:
        # Create trainable frequency parameters for each detected frequency (chord support)
        freq_params = [lv.ValueTrainableParameter(f) for f in note["frequencies"]]
        trainable_notes.append(
            {
                "frequencies": freq_params,  # List of trainable frequencies
                "start_time": lv.ValueTrainableParameter(note["start_time"]),
                "duration": lv.ValueTrainableParameter(note["duration"]),
            }
        )

    # Build instrument synthesis graph
    print(f"Building {instrument_name} synthesis graph...")
    instrument_blueprint = learnable.get_trainable_instrument(instrument_name)

    # Create time value
    time_val = lv.BasicScaling(
        value=lv.Identity(),
        mult_scale=lv.Constant(1 / sample_rate),
        sum_scale=lv.Constant(0),
    )

    # Create notes (with chord support)
    note_values = []
    for tn in trainable_notes:
        # For percussion instruments (kick, snare, hihat) that don't use frequency
        if instrument_name in ["kick", "snare", "hihat_closed", "hihat_open"]:
            note_val = instrument_blueprint(
                time=time_val, start_time=tn["start_time"].value.item()
            )
            note_values.append(note_val)
        else:
            # Create a note for each frequency in the chord
            chord_voices = []
            for freq_param in tn["frequencies"]:
                voice = instrument_blueprint(
                    time=time_val,
                    frequency=freq_param,
                    start_time=tn["start_time"].value.item(),
                    duration=tn["duration"].value.item(),
                )
                chord_voices.append(voice)

            # Sum the voices for this chord
            if len(chord_voices) == 1:
                note_values.append(chord_voices[0])
            else:
                # Average amplitude to avoid clipping
                note_values.append(
                    lv.Product(
                        lv.Sum(chord_voices), lv.Constant(1.0 / len(chord_voices))
                    )
                )

    # Sum all notes
    if len(note_values) == 1:
        synth_output = note_values[0]
    else:
        synth_output = lv.Sum(note_values)

    # Collect all trainable parameters
    print("Collecting trainable parameters...")
    all_params = collect_trainable_parameters(synth_output)

    # Add note timing/frequency parameters
    for tn in trainable_notes:
        for freq_param in tn["frequencies"]:
            all_params.append(freq_param.value)
        all_params.append(tn["start_time"].value)
        all_params.append(tn["duration"].value)

    # Remove duplicates
    all_params = list(set(all_params))

    # Enable gradients
    for param in all_params:
        param.requires_grad = True

    print(f"Total trainable parameters: {len(all_params)}")

    # Setup optimizer
    optimizer = optim.Adam(all_params, lr=learning_rate)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    history = {"losses": [], "epochs": []}

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Render audio with current parameters
        idx_buffer = torch.arange(
            len(target_tensor), dtype=torch.float32, device=device
        )
        synthesized = synth_output.getitem_torch(idx_buffer, sample_rate, device=device)

        # Compute multi-resolution spectral loss with high-frequency emphasis
        loss = multi_resolution_spectral_loss(
            synthesized,
            target_tensor,
            sample_rate=sample_rate,
            high_freq_emphasis=2.0,  # Punish high frequencies 2x more
        )

        # Backpropagation
        loss.backward()

        # Optimization step
        optimizer.step()

        # Log progress
        history["losses"].append(loss.item())
        history["epochs"].append(epoch)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Spectral Loss: {loss.item():.6f}")

    print("\n=== Training Complete ===")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save final synthesized audio
    final_idx = torch.arange(len(target_tensor), dtype=torch.float32, device=device)
    final_audio = (
        synth_output.getitem_torch(final_idx, sample_rate, device=device)
        .detach()
        .cpu()
        .numpy()
    )

    output_path = os.path.join(output_dir, f"{instrument_name}_trained.wav")
    wavfile.write(output_path, sample_rate, (final_audio * 32767).astype(np.int16))
    print(f"Saved synthesized audio to: {output_path}")

    # Save target for comparison
    target_path = os.path.join(output_dir, f"{instrument_name}_target.wav")
    wavfile.write(target_path, sample_rate, (target_audio * 32767).astype(np.int16))
    print(f"Saved target audio to: {target_path}")

    # Save parameters
    param_dict = {}
    for i, param in enumerate(all_params):
        param_dict[f"param_{i}"] = param.detach().cpu().item()

    import json

    param_path = os.path.join(output_dir, f"{instrument_name}_params.json")
    with open(param_path, "w") as f:
        json.dump(param_dict, f, indent=2)
    print(f"Saved parameters to: {param_path}")

    return history


#
### MAIN ###
#


def main():
    parser = argparse.ArgumentParser(
        description="Train synthesized instruments from WAV files"
    )

    parser.add_argument("wav_file", type=str, help="Path to input WAV file")
    parser.add_argument(
        "--instrument",
        "-i",
        type=str,
        default="sine",
        choices=list(learnable.TRAINABLE_INSTRUMENTS.keys()),
        help="Instrument type to train",
    )
    parser.add_argument(
        "--num-notes", "-n", type=int, default=1, help="Number of notes in the segment"
    )
    parser.add_argument(
        "--start", "-s", type=float, default=0.0, help="Start time of segment (seconds)"
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=5.0,
        help="Duration of segment (seconds)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--sample-rate", "-r", type=int, default=44100, help="Sample rate"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="trained_models",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu, cuda, cuda:0, etc.)",
    )

    args = parser.parse_args()

    # Check if WAV file exists
    if not os.path.exists(args.wav_file):
        print(f"Error: WAV file not found: {args.wav_file}")
        return

    # Train
    history = train_instrument(
        wav_path=args.wav_file,
        instrument_name=args.instrument,
        num_notes=args.num_notes,
        segment_start=args.start,
        segment_duration=args.duration,
        sample_rate=args.sample_rate,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("\nTraining history available in returned dict.")
    print(f"Final loss: {history['losses'][-1]:.6f}")


if __name__ == "__main__":
    main()

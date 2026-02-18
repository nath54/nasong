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


"""Training configuration and hyperparameter management.

This module defines the configuration structure for instrument training,
including audio settings, note detection parameters, loss functions, and
training engine options. It uses dataclasses for type safety and provides
YAML serialization.
"""

#
### Import Modules. ###
#
import dataclasses
from dataclasses import dataclass, field

#
import yaml  # type: ignore


@dataclass
class AudioConfig:
    """Configuration for target audio loading and sampling.

    Attributes:
        start_time (float): The start offset in the WAV file (seconds).
        duration (float): The length of the segment to load (seconds).
        sample_rate (int): The target sample rate for processing (Hz).
    """

    start_time: float = 0.0
    duration: float = 5.0
    sample_rate: int = 44100


@dataclass
class NoteDetectionConfig:
    """Settings for the note detection / transcription preprocessing.

    Attributes:
        method (str): The detection backend to use (e.g., 'legacy', 'onnx_crepe').
        onnx_path (str): Path to the ONNX model weights (if applicable).
    """

    method: str = "legacy"  # 'legacy', 'basic_pitch_onnx', 'librosa', 'torchcrepe'
    onnx_path: str = "models/nmp.onnx"

    # Basic Pitch Params
    bp_onset_threshold: float = 0.5
    bp_frame_threshold: float = 0.3
    bp_min_note_len: float = 58.0  # ms
    bp_min_freq: float = 50.0
    bp_max_freq: float = 2000.0

    # Legacy Params
    legacy_frame_len: float = 0.02
    legacy_hop_len: float = 0.01
    legacy_onset_threshold: float = 0.3
    legacy_num_notes: int = 1
    legacy_use_onset: bool = True
    legacy_max_pitches: int = 3
    legacy_min_freq: float = 50.0
    legacy_max_freq: float = 4000.0

    # Librosa Params
    librosa_fmin: float = 50.0
    librosa_fmax: float = 2000.0
    librosa_frame_length: int = 2048
    librosa_hop_length: int = 512

    # TorchCrepe Params
    crepe_model: str = "full"  # tiny, small, medium, large, full
    crepe_step_size: int = 10  # ms
    crepe_confidence_threshold: float = 0.8

    # AudioFlux Params
    audioflux_type: str = "PEF"  # PEF, YIN
    audioflux_min_freq: float = 50.0
    audioflux_max_freq: float = 2000.0
    audioflux_slide_length: int = 1024
    audioflux_radix2_exp: int = 12


@dataclass
class SpectralLossConfig:
    """Parameters for the Multi-Resolution STFT Spectral Loss.

    Attributes:
        n_fft (int): Standard FFT size.
        hop_length (int): Distance between STFT windows.
        high_freq_emphasis (float): Multiplier for high-frequency loss weight.
        fft_sizes (list[int]): List of FFT sizes for multi-resolution analysis.
    """

    n_fft: int = 2048
    hop_length: int = 512
    high_freq_emphasis: float = 2.0
    fft_sizes: list[int] = field(default_factory=lambda: [2048, 1024, 512])


@dataclass
class TrainingConfig:
    """Global configuration for an instrument training experiment.

    Attributes:
        instrument_name (str): Identifier for the instrument blueprint.
        target_wav (str): Path to the reference audio file.
        output_dir (str): Base directory for saving model parameters and audio.
        device (str): Execution device ('cpu' or 'cuda').
        engine_type (str): Gradient engine ('autograd', 'numpy', or 'torch').
        epochs (int): Number of optimization iterations.
        learning_rate (float): Step size for the optimizer.
        save_interval (int): How often (in epochs) to save intermediate results.
        audio (AudioConfig): Specific audio loading settings.
        note_detection (NoteDetectionConfig): Settings for note extraction.
        spectral_loss (SpectralLossConfig): Loss function hyperparameters.
        train_duration (float): segment length for main training split.
        val_duration (float): segment length for validation split.
        test_duration (float): segment length for final testing.
        batch_duration (float): Maximum audio length per training step.
        batch_overlap (float): Overlap between batches in seconds.
        save_config (bool): Whether to save this config to YAML in the output.
        save_history (bool): Whether to save loss history to JSON in the output.
    """

    instrument_name: str
    target_wav: str
    output_dir: str = "trained_models"
    device: str = "cpu"
    engine_type: str = "autograd"  # 'autograd', 'numpy', 'torch'
    epochs: int = 100
    learning_rate: float = 0.01
    save_interval: int = 10

    audio: AudioConfig = field(default_factory=AudioConfig)
    note_detection: NoteDetectionConfig = field(default_factory=NoteDetectionConfig)
    spectral_loss: SpectralLossConfig = field(default_factory=SpectralLossConfig)

    # Splitting and Batching
    train_duration: float = 10.0  # Duration to use for training
    val_duration: float = 5.0  # Duration for validation
    test_duration: float = 5.0  # Duration for testing
    batch_duration: float = 5.0  # Max duration per training batch
    batch_overlap: float = 1.0  # overlap in seconds
    save_config: bool = True
    save_history: bool = True

    @staticmethod
    def from_yaml(path: str) -> "TrainingConfig":
        """Loads configuration from a YAML file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            TrainingConfig: The loaded configuration object.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Recursive helper to load nested dataclasses
        def load_dataclass(cls, data_dict):
            if data_dict is None:
                return cls()

            field_names = {f.name for f in dataclasses.fields(cls)}
            filtered_data = {k: v for k, v in data_dict.items() if k in field_names}

            # Handle nested dataclasses manually
            for field_obj in dataclasses.fields(cls):
                if dataclasses.is_dataclass(field_obj.type):
                    config_dict = filtered_data.get(field_obj.name, {})
                    filtered_data[field_obj.name] = load_dataclass(
                        field_obj.type, config_dict
                    )

            return cls(**filtered_data)

        return load_dataclass(TrainingConfig, data)

    def to_yaml(self, path: str) -> None:
        """Saves the current configuration to a YAML file.

        Args:
            path (str): Output file path.
        """

        def as_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: as_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(as_dict(self), f, default_flow_style=False)

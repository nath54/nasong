from dataclasses import dataclass, field
import dataclasses
from typing import List
import yaml


@dataclass
class AudioConfig:
    start_time: float = 0.0
    duration: float = 5.0
    sample_rate: int = 44100


@dataclass
class NoteDetectionConfig:
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
    n_fft: int = 2048
    hop_length: int = 512
    high_freq_emphasis: float = 2.0
    fft_sizes: List[int] = field(default_factory=lambda: [2048, 1024, 512])


@dataclass
class TrainingConfig:
    instrument_name: str
    target_wav: str
    output_dir: str = "trained_models"
    device: str = "cpu"
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
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
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

    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""

        def as_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: as_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(as_dict(self), f, default_flow_style=False)

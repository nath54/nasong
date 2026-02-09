# Training Configuration Documentation

The NaSong training pipeline is configured using YAML files. This allows for reproducible experiments and easy tuning of hyperparameters without modifying the code.

## Configuration Structure

The configuration is divided into three main sections:
1.  **Global Training Parameters**: General settings like epochs, learning rate, device.
2.  **Audio Settings**: Parameters for loading and processing the input audio.
3.  **Note Detection**: Settings for the note detection algorithm.
4.  **Spectral Loss**: Parameters for the loss function.

### Global Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `instrument_name` | string | (Required) | Name of the instrument to train (e.g., 'sine', 'saw', 'violin'). |
| `target_wav` | string | (Required) | Path to the target WAV file. |
| `output_dir` | string | "trained_models" | Directory to save trained models and audio. |
| `device` | string | "cpu" | Device to use ("cpu", "cuda", "mps"). |
| `epochs` | int | 100 | Number of training epochs. |
| `learning_rate` | float | 0.01 | Learning rate for Adam optimizer. |

### Audio Settings (`audio`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `start_time` | float | 0.0 | Start time of the training segment (seconds). |
| `duration` | float | 5.0 | Duration of the training segment (seconds). |
| `sample_rate` | int | 44100 | Sample rate to use for training. |

### Note Detection (`note_detection`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `method` | string | "legacy" | Algorithm to use: `legacy`, `basic_pitch_onnx`, `librosa`, `torchcrepe`. |
| `onnx_path` | string | "models/nmp.onnx" | Path to Basic Pitch ONNX model (for `basic_pitch_onnx`). |

#### Method-Specific Parameters

**Basic Pitch (`basic_pitch_onnx`, or alias `neural_note`)**
- `bp_onset_threshold`: (0.5) Sensitivity for note onsets.
- `bp_frame_threshold`: (0.3) Confidence threshold for active frames.
- `bp_min_note_len`: (58.0) Minimum note length in ms.
- `bp_min_freq`: (50.0) Minimum frequency in Hz.
- `bp_max_freq`: (2000.0) Maximum frequency in Hz.
- Note: The `neural_note` method is an alias for `basic_pitch_onnx` as NeuralNote is based on the Basic Pitch ONNX model.

**AudioFlux (`audioflux`)**
- `audioflux_type`: ("PEF") Algorithm type: `PEF` (Pitch Estimation Filter) or `YIN`.
- `audioflux_min_freq`: (50.0) Minimum frequency.
- `audioflux_max_freq`: (2000.0) Maximum frequency.
- `audioflux_slide_length`: (1024) Slide length.
- `audioflux_radix2_exp`: (12) Radix-2 exponent for FFT size (2^12 = 4096).

**TorchCrepe (`torchcrepe`)**
- `crepe_model`: ("medium") Model size: tiny, small, medium, large, full.
- `crepe_step_size`: (10) Step size in ms.
- `crepe_confidence_threshold`: (0.8) Confidence threshold for voicing (0-1).

**Librosa (`librosa`)**
- `librosa_fmin`: (50.0) Minimum frequency.
- `librosa_fmax`: (2000.0) Maximum frequency.
- `librosa_frame_length`: (2048) STFT frame length.

**Legacy (`legacy`)**
- `legacy_onset_threshold`: (0.3) Relative energy threshold for onsets.
- `legacy_frame_len`: (0.02) Frame length in seconds for energy calculation.

### Spectral Loss (`spectral_loss`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fft_sizes` | list[int] | [2048, 1024, 512] | List of FFT sizes for multi-resolution loss. |
| `high_freq_emphasis` | float | 2.0 | Weight multiplier for high frequencies. |

## Example Usage

Run training with a config file:
```bash
python -m nasong.trainable.train --config training_configs/my_experiment.yaml
```

Override parameters from CLI:
```bash
python -m nasong.trainable.train --config training_configs/base.yaml --epochs 500 --lr 0.05
```

## Dependencies for Note Detection

Depending on the selected method, you may need to install additional packages:

- **Legacy**: No extra dependencies.
- **Librosa**:
  ```bash
  pip install librosa
  ```
- **TorchCrepe**:
  ```bash
  pip install torchcrepe
  ```
- **AudioFlux**:
  ```bash
  pip install audioflux
  ```
- **Crepe ONNX**:
  ```bash
  pip install onnxruntime
  ```

### Notes on Other Libraries
- **Omnizart**: This library was evaluated but is currently not supported due to its heavy reliance on outdated dependencies (Librosa < 0.9, Python < 3.12) which would conflict with modern environments.

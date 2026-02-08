from typing import List, Dict, Any
import numpy as np

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None

from .base import NoteDetector

try:
    import torchcrepe
except ImportError:
    torchcrepe = None


class TorchCrepeDetector(NoteDetector):
    """
    Note detection using TorchCrepe (viterbi decoding + segmentation).
    High accuracy for monophonic audio.
    """

    def detect(
        self, audio_segment: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Cannot use TorchCrepe.")

        if torchcrepe is None:
            raise ImportError(
                "TorchCrepe is not installed. Please install 'torchcrepe'."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prepare audio tensor
        # Crepe expects shape (batch, time)
        audio_tensor = torch.tensor(
            audio_segment, dtype=torch.float32, device=device
        ).unsqueeze(0)

        step_size_ms = self.config.get("crepe_step_size", 10)
        hop_length = int(step_size_ms * sample_rate / 1000)

        fmin = 50.0
        fmax = 2000.0
        model_size = self.config.get("crepe_model", "medium")
        conf_thresh = self.config.get("crepe_confidence_threshold", 0.8)

        # Predict
        try:
            # Note: predict returns pitch in Hz and periodicity (confidence)
            pitch, periodicity = torchcrepe.predict(
                audio_tensor,
                sample_rate,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                model=model_size,
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
                device=device,
                batch_size=2048,
            )
        except Exception as e:
            raise RuntimeError(f"TorchCrepe prediction failed: {e}")

        # Move to CPU for processing
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()

        # Segmentation logic
        # Find continuous regions where periodicity > threshold
        is_voiced = periodicity > conf_thresh

        notes = []
        hop_s = step_size_ms / 1000.0

        current_start_idx = None
        current_pitches = []
        current_confs = []

        for i, voiced in enumerate(is_voiced):
            if voiced:
                if current_start_idx is None:
                    current_start_idx = i
                current_pitches.append(pitch[i])
                current_confs.append(periodicity[i])
            else:
                if current_start_idx is not None:
                    # End of note
                    self._add_note(
                        notes,
                        current_start_idx,
                        i,
                        current_pitches,
                        current_confs,
                        hop_s,
                        audio_segment,
                        sample_rate,
                    )
                    current_start_idx = None
                    current_pitches = []
                    current_confs = []

        # Check last note
        if current_start_idx is not None:
            self._add_note(
                notes,
                current_start_idx,
                len(is_voiced),
                current_pitches,
                current_confs,
                hop_s,
                audio_segment,
                sample_rate,
            )

        return notes

    def _add_note(
        self,
        notes,
        start_idx,
        end_idx,
        pitches,
        confs,
        hop_s,
        audio_segment,
        sample_rate,
    ):
        duration = (end_idx - start_idx) * hop_s
        # Min duration, e.g. 50ms
        if duration < 0.05:
            return

        start_time = start_idx * hop_s
        end_time = end_idx * hop_s

        # Calculate amplitude (RMS)
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Clamp indices
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_segment), end_sample)

        if end_sample > start_sample:
            segment = audio_segment[start_sample:end_sample]
            amplitude = float(np.sqrt(np.mean(segment**2)))
        else:
            amplitude = 0.0

        # Median pitch to robustly handle fluctuations
        median_pitch = np.median(pitches)
        mean_conf = np.mean(confs)

        notes.append(
            {
                "start_time": float(start_time),
                "duration": float(duration),
                "frequencies": [float(median_pitch)],
                "confidence": float(mean_conf),
                "amplitude": amplitude,
            }
        )

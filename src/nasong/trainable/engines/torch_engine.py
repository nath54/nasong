#
### PyTorch Training Engine. ###
#

from typing import Dict, List, Any, Optional, Set

try:
    import torch
    import torch.optim as optim
    from torch import Tensor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = Any
    optim = Any

    class Tensor:
        pass


import numpy as np
from numpy.typing import NDArray

# Internal imports
from nasong.trainable.engines.base import BaseTrainingEngine
from nasong.core.value import Value, ValueTrainableParameter


class TorchEngine(BaseTrainingEngine):
    """
    Training engine using PyTorch for automatic differentiation.
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the Torch engine.

        Args:
            config: TrainingConfig object containing hyperparameters.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not available. execution stopped.")

        super().__init__(config)
        self.device: str = getattr(config, "device", "cpu")
        self.optimizer: Optional[optim.Optimizer] = None
        self.all_params: List[Tensor] = []

    def spectral_loss(
        self,
        synthesized: Tensor,
        target: Tensor,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        high_freq_emphasis: float = 2.0,
    ) -> Tensor:
        """PyTorch implementation of spectral loss."""
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

        return mag_loss + 0.5 * log_mag_loss

    def multi_resolution_spectral_loss(
        self,
        synthesized: Tensor,
        target: Tensor,
        sample_rate: int = 44100,
        fft_sizes: Optional[List[int]] = None,
        high_freq_emphasis: float = 2.0,
    ) -> Tensor:
        """PyTorch implementation of multi-resolution spectral loss."""
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512]

        total_loss: Tensor = torch.tensor(0.0, device=self.device)

        for n_fft in fft_sizes:
            hop_length = n_fft // 4
            loss = self.spectral_loss(
                synthesized, target, sample_rate, n_fft, hop_length, high_freq_emphasis
            )
            total_loss = total_loss + loss

        return total_loss / len(fft_sizes)

    def collect_trainable_parameters(
        self, value: Value, params: Optional[Set[Tensor]] = None
    ) -> List[Tensor]:
        """Recursively collects all Torch tensors that require gradients."""
        if params is None:
            params = set()

        if isinstance(value, ValueTrainableParameter):
            if isinstance(value.value, Tensor):
                params.add(value.value)

        # Explore children
        for attr_name in dir(value):
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(value, attr_name)
                if isinstance(attr, Value):
                    self.collect_trainable_parameters(attr, params)
                elif isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, Value):
                            self.collect_trainable_parameters(item, params)
                        elif isinstance(item, tuple):
                            for sub_item in item:
                                if isinstance(sub_item, Value):
                                    self.collect_trainable_parameters(sub_item, params)
            except Exception:
                continue

        return list(params)

    def initialize_optimizer(self, blueprint: Value) -> None:
        """Initializes the optimizer with parameters from the graph."""
        self.all_params = self.collect_trainable_parameters(blueprint)
        for p in self.all_params:
            p.requires_grad = True

        lr = getattr(self.config, "learning_rate", 0.01)
        self.optimizer = optim.Adam(self.all_params, lr=lr)

    def compute_loss(
        self, target_audio: NDArray[np.float32], blueprint: Value, sample_rate: int
    ) -> float:
        """Currently unimplemented as a standalone atomic op, usually handled in batch loops."""
        # For TorchEngine, we typically do this inside the batch loop in `step`
        # or a higher-level loop. But we'll provide a basic implementation.
        target_tensor = torch.from_numpy(target_audio).to(self.device).float()

        # Identity indices for full rendering (careful with memory)
        indices = torch.arange(len(target_audio), device=self.device).float()
        synthesized = blueprint.getitem_torch(indices, sample_rate, device=self.device)

        loss = self.multi_resolution_spectral_loss(
            synthesized,
            target_tensor,
            sample_rate,
            fft_sizes=getattr(self.config.spectral_loss, "fft_sizes", None),
            high_freq_emphasis=getattr(
                self.config.spectral_loss, "high_freq_emphasis", 2.0
            ),
        )

        # We don't call .backward() here to allow standard compute_loss calls for val/test
        return loss.item()

    def step(self) -> Dict[str, float]:
        """Placeholder for a single step. Real world uses batch loops."""
        # In a real training task, the batch loop handles the .backward() calls.
        # This interface might need refinement if we want the engine to own the loop.
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return {}

    def get_parameter_values(self) -> Dict[str, float]:
        """Returns name:value dictionary of parameters."""
        # Note: Parameters might not have names if they were captured generically.
        # We might need a better way to map names back if we want to save/restore by name.
        results = {}
        for i, p in enumerate(self.all_params):
            results[f"param_{i}"] = p.detach().cpu().item()
        return results

    def set_parameter_values(self, parameters: Dict[str, float]) -> None:
        """Sets tensor values from a dictionary."""
        # This requires a naming convention consistency.
        # For now, we assume simple index-based or name-based if available.
        # This will be refined as we implement model saving/loading.
        pass

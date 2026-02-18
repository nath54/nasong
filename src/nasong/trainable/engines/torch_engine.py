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


"""Optimization engine using PyTorch for automatic differentiation.

This engine provides the fastest training by leveraging PyTorch tensors and
GPU acceleration (if available). It converts Value nodes to Torch-compatible
computations and uses standard PyTorch optimizers.
"""

#
### Import Modules. ###
#
from typing import Any, Optional

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.trainable.engines.base import BaseTrainingEngine
from nasong.core.value import Value, ValueTrainableParameter

#
### Try to import torch and torch.optim and torch.Tensor. ###
#
try:
    import torch
    import torch.optim as optim
    from torch import Tensor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None  # type: ignore
    optim = None  # type: ignore

    class Tensor:  # type: ignore[no-redef] # pylint: disable=too-few-public-methods
        """
        Mock Tensor class for type hinting when torch is not available.
        """

        pass  # pylint: disable=unnecessary-pass


#
class TorchEngine(BaseTrainingEngine):
    """Training engine using PyTorch for automatic differentiation.

    Automates the rendering of audio graphs as PyTorch tensors and uses
    Torch's native autograd for gradient calculation. Particularly efficient
    for multi-resolution spectral loss.

    Attributes:
        device (str): Computation device ('cpu', 'cuda').
        optimizer (Optional[optim.Optimizer]): The Torch optimizer instance.
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
        self.all_params: list[Tensor] = []

    def spectral_loss(
        self,
        synthesized: Tensor,
        target: Tensor,
        sample_rate: int = 44100,  # pylint: disable=unused-argument
        n_fft: int = 2048,
        hop_length: int = 512,
        high_freq_emphasis: float = 2.0,
    ) -> Tensor:
        """PyTorch calculation of spectral convergence and log-magnitude distance.

        Args:
            synthesized (Tensor): The generated audio tensor.
            target (Tensor): The ground truth audio tensor.
            sample_rate (int): Sampling rate in Hz.
            n_fft (int): Size of the FFT window.
            hop_length (int): Distance between windows.
            high_freq_emphasis (float): Weight for high frequencies.

        Returns:
            Tensor: Scalar loss value.
        """
        synth_stft = torch.stft(  # pylint: disable=no-member
            synthesized,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=synthesized.device),
            return_complex=True,
        )
        target_stft = torch.stft(  # pylint: disable=no-member
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=target.device),
            return_complex=True,
        )

        synth_mag = torch.abs(synth_stft)  # pylint: disable=no-member
        target_mag = torch.abs(target_stft)  # pylint: disable=no-member

        freq_bins = synth_mag.shape[0]
        freq_weights = torch.linspace(  # pylint: disable=no-member
            1.0, high_freq_emphasis, freq_bins, device=synthesized.device
        )
        freq_weights = freq_weights.unsqueeze(1)

        synth_mag_weighted = synth_mag * freq_weights
        target_mag_weighted = target_mag * freq_weights

        mag_loss = torch.mean(torch.abs(synth_mag_weighted - target_mag_weighted))  # pylint: disable=no-member

        synth_log_mag = torch.log(synth_mag + 1e-5)  # pylint: disable=no-member
        target_log_mag = torch.log(target_mag + 1e-5)  # pylint: disable=no-member
        log_mag_loss = torch.mean(torch.abs(synth_log_mag - target_log_mag))  # pylint: disable=no-member

        return mag_loss + 0.5 * log_mag_loss

    def multi_resolution_spectral_loss(
        self,
        synthesized: Tensor,
        target: Tensor,
        sample_rate: int = 44100,
        fft_sizes: Optional[list[int]] = None,
        high_freq_emphasis: float = 2.0,
    ) -> Tensor:
        """Computes a sum of spectral losses across multiple FFT resolutions.

        Args:
            synthesized (Tensor): The generated audio tensor.
            target (Tensor): The reference audio tensor.
            sample_rate (int): Sampling rate in Hz.
            fft_sizes (Optional[list[int]]): List of window sizes for analysis.
            high_freq_emphasis (float): High-frequency boost.

        Returns:
            Tensor: Averaged multi-resolution loss.
        """
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512]

        total_loss: Tensor = torch.tensor(0.0, device=self.device)  # pylint: disable=no-member

        for n_fft in fft_sizes:
            hop_length = n_fft // 4
            loss = self.spectral_loss(
                synthesized, target, sample_rate, n_fft, hop_length, high_freq_emphasis
            )
            total_loss = total_loss + loss

        return total_loss / len(fft_sizes)

    def collect_trainable_parameters(
        self, value: Value, params: Optional[set[Tensor]] = None
    ) -> list[Tensor]:
        """Recursively scans the Value graph for Torch tensors requiring gradients.

        Args:
            value (Value): The starting node of the graph.
            params (Optional[set[Tensor]]): Accumulator set for unique tensors.

        Returns:
            list[Tensor]: A list of all unique trainable tensors found.
        """
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
            except Exception:  # pylint: disable=broad-except
                continue

        return list(params)

    def initialize_optimizer(self, blueprint: Value) -> None:
        """Collects parameters from the audio graph and initializes the Adam optimizer.

        Args:
            blueprint (Value): The root of the synthesis graph.
        """
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
        target_tensor = torch.from_numpy(target_audio).to(self.device).float()  # pylint: disable=no-member

        # Identity indices for full rendering (careful with memory)
        indices = torch.arange(len(target_audio), device=self.device).float()  # pylint: disable=no-member
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

    def step(self) -> dict[str, float]:
        """Placeholder for a single step. Real world uses batch loops."""
        # In a real training task, the batch loop handles the .backward() calls.
        # This interface might need refinement if we want the engine to own the loop.
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return {}

    def get_parameter_values(self) -> dict[str, float]:
        """Returns name:value dictionary of parameters."""
        # Note: Parameters might not have names if they were captured generically.
        # We might need a better way to map names back if we want to save/restore by name.
        results = {}
        for i, p in enumerate(self.all_params):
            results[f"param_{i}"] = p.detach().cpu().item()
        return results

    def set_parameter_values(self, parameters: dict[str, float]) -> None:
        """Sets tensor values from a dictionary."""
        # This requires a naming convention consistency.
        # For now, we assume simple index-based or name-based if available.
        # This will be refined as we implement model saving/loading.
        pass  # pylint: disable=unnecessary-pass

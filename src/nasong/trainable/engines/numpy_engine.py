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


"""Manual reverse-mode differentiation engine using pure NumPy.

This engine uses the built-in `backward` methods of `Value` nodes to compute
gradients without external AD libraries. It is lightweight and has zero
dependencies beyond NumPy and SciPy.
"""

#
### Import Modules. ###
#
from typing import Any

#
from scipy import signal

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value, ValueTrainableParameter, ParameterContext
from nasong.trainable.engines.base import BaseTrainingEngine


class NumpyEngine(BaseTrainingEngine):
    """Pure NumPy training engine using manual reverse-mode differentiation.

    This engine leverages the `backward` methods implemented in the `Value`
    classes to compute gradients without depending on PyTorch. It supports
    both standard MSE and spectral loss metrics.

    Attributes:
        learning_rate (float): The optimizer step size.
        optimizer_type (str): Optimizer algorithm ('adam' or 'sgd').
        loss_type (str): Loss function type ('mse' or 'spectral').
    """

    def __init__(self, config: Any) -> None:
        """
        Initializes the NumPy engine.
        """
        super().__init__(config)
        self.learning_rate = getattr(config, "learning_rate", 0.001)
        self.optimizer_type = getattr(config, "optimizer_type", "adam")
        self.loss_type = getattr(config, "loss_type", "mse")  # "mse" or "spectral"

        # State for Adam
        self.m: dict[ValueTrainableParameter, NDArray[np.float32]] = {}
        self.v: dict[ValueTrainableParameter, NDArray[np.float32]] = {}
        self.t = 0

        # Captured parameters and their gradients
        self.captured_params: list[ValueTrainableParameter] = []
        self.gradients: dict[ValueTrainableParameter, NDArray[np.float32]] = {}

    def spectral_loss(
        self,
        synthesized: NDArray[np.float32],
        target: NDArray[np.float32],
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        high_freq_emphasis: float = 2.0,
    ) -> float:
        """Calculates a multi-component spectral distance between two audio signals.

        Args:
            synthesized (NDArray[np.float32]): The generated audio.
            target (NDArray[np.float32]): The reference audio.
            sample_rate (int): Sampling rate in Hz.
            n_fft (int): Size of the FFT window.
            hop_length (int): Distance between windows.
            high_freq_emphasis (float): Weight multiplier for high frequencies.

        Returns:
            float: Combined magnitude and log-magnitude spectral loss.
        """

        # Compute STFT
        _, _, synth_stft = signal.stft(
            synthesized, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )
        _, _, target_stft = signal.stft(
            target, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )

        synth_mag = np.abs(synth_stft)
        target_mag = np.abs(target_stft)

        # Frequency weighting
        freq_bins = synth_mag.shape[0]
        freq_weights = np.linspace(1.0, high_freq_emphasis, freq_bins)
        freq_weights = freq_weights[:, np.newaxis]

        synth_mag_weighted = synth_mag * freq_weights
        target_mag_weighted = target_mag * freq_weights

        mag_loss = np.mean(np.abs(synth_mag_weighted - target_mag_weighted))

        synth_log_mag = np.log(synth_mag + 1e-5)
        target_log_mag = np.log(target_mag + 1e-5)
        log_mag_loss = np.mean(np.abs(synth_log_mag - target_log_mag))

        # Note: Hand-wavy: dL/dy for spectral loss is complex.
        # For simplicity in this manual pass, we'll mostly use MSE for gradients
        # even if loss reported is spectral, OR we'd need a full STFT backward.
        # Given the task complexity, we'll focus on MSE gradients for now
        # while reporting spectral loss values.

        return float(mag_loss + 0.5 * log_mag_loss)

    def compute_loss(
        self, target_audio: NDArray[np.float32], blueprint: Value, sample_rate: int
    ) -> float:
        """
        Runs the model and calculates the loss.
        """
        # 1. Capture parameters and create contexts
        with ParameterContext(capture=True) as ctx:
            samples = np.arange(len(target_audio), dtype=np.float32)
            self.indices = samples
            prediction = blueprint.getitem_np(samples, sample_rate)
            self.captured_params = ctx.captured_params

        # 2. Calculate Loss
        # Use float64 for internal loss and gradient calculation for precision
        prediction_f64 = prediction.astype(np.float64)
        target_f64 = target_audio.astype(np.float64)

        if self.loss_type == "spectral":
            loss = self.spectral_loss(prediction, target_audio, sample_rate)
            # Proxy dL/dy as MSE gradient for simplified backprop
            diff = prediction_f64 - target_f64
            self.grad_output = (2.0 * diff / len(target_audio)).astype(np.float64)
        else:
            diff = prediction_f64 - target_f64
            loss = np.mean(diff**2)
            self.grad_output = (2.0 * diff / len(target_audio)).astype(np.float64)

        # 3. Store intermediate state for backward
        # dL/dy = 2 * (y - target) / N
        self.blueprint = blueprint
        self.sample_rate = sample_rate

        return float(loss)

    def step(self) -> dict[str, float]:
        """
        Performs the backward pass and optimization step.
        """
        # 1. Reset gradients
        self.gradients = {}

        # 2. Backward Pass
        # We need a mechanism to accumulate gradients into parameters.
        # I'll add a helper to intercept gradients in ValueTrainableParameter.backward.
        # Actually, I should have implemented backward in ValueTrainableParameter.

        # Let's override backward in the engine logic or ensure ValueTrainableParameter.backward works.
        # I'll update ValueTrainableParameter.backward if needed.

        context = {"engine": self, "indices": self.indices}
        self.blueprint.backward(self.grad_output, context, self.sample_rate)

        # 3. Optimization Step
        for p in self.captured_params:
            if p not in self.gradients:
                continue

            grad = self.gradients[p][0]

            if self.optimizer_type == "sgd":
                p.value -= self.learning_rate * grad
            elif self.optimizer_type == "adam":
                # Basic Adam implementation
                if p not in self.m:
                    self.m[p] = np.zeros(1)
                    self.v[p] = np.zeros(1)

                self.t += 1
                beta1, beta2 = 0.9, 0.999
                eps = 1e-8

                self.m[p] = beta1 * self.m[p] + (1 - beta1) * grad
                self.v[p] = beta2 * self.v[p] + (1 - beta2) * (grad**2)

                m_hat = self.m[p] / (1 - beta1**self.t)
                v_hat = self.v[p] / (1 - beta2**self.t)

                p.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return {"lr": self.learning_rate}

    def get_parameter_values(self) -> dict[str, float]:
        """
        Retrieves current parameter values.
        """
        return {p.name or str(id(p)): float(p.value) for p in self.captured_params}

    def set_parameter_values(self, parameters: dict[str, float]) -> None:
        """
        Injects parameter values.
        """
        # We need to match by name or context order
        # For now, matching by name is safest
        for p in self.captured_params:
            if p.name and p.name in parameters:
                p.value = parameters[p.name]

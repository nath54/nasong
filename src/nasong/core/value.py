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


"""
Fundamental Value abstraction for audio generation.

This module defines the `Value` base class and related infrastructure for
time-varying signals. It includes support for vectorized operations (NumPy/PyTorch),
automatic differentiation (PyTorch), and manual differentiation (NumPy). It also
provides a mechanism for trainable parameters through `ValueTrainableParameter`
and `ParameterContext`.
"""

#
### Import Modules. ###
#
from typing import Any

#
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.values.mult_itms_ops.value_sum import Sum
from nasong.core.values.basic.value_constant import Constant
from nasong.core.values.mult_itms_ops.value_product import Product
from nasong.core.values.complex.value_pow import Pow
from nasong.core.values.single_itms_ops.value_modulo import Modulo

#
try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = Any  # type: ignore # Mock for imports

    class Tensor:  # pylint: disable=missing-class-docstring
        pass  # Mock for runtime type hints


#
### ABSTRACT CLASS. ###
#


#
class Value:
    """Abstract base class for a time-varying value.

    This class defines the interface for all 'Value' objects, which are used
    to generate signals, envelopes, modulations, etc., on a per-sample basis.
    Subclasses must implement at least one of the `get_item`, `getitem_np`, or
    `getitem_torch` methods.
    """

    #
    def __init__(self) -> None:
        """Initializes the base Value object."""

        #
        pass  # pylint: disable=unnecessary-pass

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Calculates the value at a single sample index.

        This is the non-vectorized, sample-by-sample method. It is primarily
        used as a fallback or for simple scalar calculations.

        Args:
            index (int): The discrete sample index.
            sample_rate (int): The audio sample rate in Hz.

        Returns:
            float: The calculated value at the given index.
        """

        print(f"Value.get_item({index}, {sample_rate})")

        #
        return 0

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Calculates values for an array of sample indexes using NumPy.

        This is the performance-critical method used for rendering audio blocks
        using the CPU. Subclasses should override this with a fast, vectorized
        NumPy implementation.

        Args:
            indexes_buffer (NDArray[np.float32]): A NumPy array of sample indexes.
            sample_rate (int): The audio sample rate in Hz.

        Returns:
            NDArray[np.float32]: A NumPy array of calculated values, matching the
                shape of `indexes_buffer`.
        """

        #
        ### If we arrive here, ###
        ### it is because there are not implemented getitem_np method, ###
        ### so we are using this non optimized placeholder. ###
        #
        default: NDArray[np.float32] = np.zeros_like(indexes_buffer, dtype=np.float32)

        #
        for idx, i in enumerate(indexes_buffer):
            #
            default[idx] = self.get_item(index=int(i), sample_rate=sample_rate)

        #
        return default

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,  # pylint: disable=unused-argument
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Calculates values for a tensor of sample indexes using PyTorch.

        This method is used for hardware-accelerated rendering and is essential
        for gradient-based parameter optimization. Implementations should focus
        on differentiable operations.

        Args:
            indexes_buffer (Tensor): A PyTorch tensor of sample indexes.
            sample_rate (int): The audio sample rate in Hz.
            device (str | torch.device, optional): The device to use for
                tensor operations. Defaults to "cpu".

        Returns:
            Tensor: A PyTorch tensor of calculated values, matching the shape
                of `indexes_buffer`.
        """

        #
        ### If we arrive here, it is because there are not implemented getitem_torch method, so we are using this non optimized placeholder. ###
        #
        default: Tensor = torch.zeros_like(
            indexes_buffer, dtype=torch.float32, device=device
        )

        #
        ### We don't use the __get_item__ method to avoid gradient discontinuity. ###
        #
        return default

    #
    #
    # === Operator Overloading ===
    #

    def __add__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([self, other])

    def __radd__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([other, self])

    def __mul__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Product([self, other])

    def __rmul__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Product([other, self])

    def __sub__(self, other):

        # self - other = self + (other * -1)
        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([self, other * Constant(-1.0)])

    def __rsub__(self, other):

        # other - self = other + (self * -1)
        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([other, self * Constant(-1.0)])

    def __truediv__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        # self / other = self * (other ** -1)
        return Product([self, Pow(other, Constant(-1.0))])

    def __rtruediv__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Product([other, Pow(self, Constant(-1.0))])

    def __mod__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Modulo(self, other)

    def __rmod__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Modulo(other, self)

    def __pow__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Pow(self, other)

    def __rpow__(self, other):

        if not isinstance(other, Value):
            other = Constant(other)
        return Pow(other, self)

    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Calculates gradients for the NumPy engine (manual differentiation).

        This method facilitates backpropagation for the non-PyTorch engine.
        Subclasses that house trainable parameters must override this to
        propagate gradients to their inputs or update internal state.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the loss with
                respect to this node's output.
            context (dict[str, Any]): Intermediate values stored during the
                forward pass (`getitem_np`).
            sample_rate (int): The audio sample rate in Hz.
        """
        pass  # pylint: disable=unnecessary-pass


#
### CONTEXT MANAGER FOR TRAINABLE PARAMETERS ###
#


#
class ParameterContext:
    """Context manager for managing `ValueTrainableParameter` instances.

    This class serves as a global (or thread-local) registry to either capture
    newly created parameters (during training) or inject specific values into
    parameters (during inference).
    """

    _current = None

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        capture: bool = False,
        ignore_unknown: bool = True,
    ):
        """
        Initialize the context manager.
        """
        self.parameters = parameters or {}
        self.capture = capture
        self.captured_params: list["ValueTrainableParameter"] = []
        self.ignore_unknown = ignore_unknown
        self._param_counter = 0

    def __enter__(self):
        """
        Enter the context manager.
        """
        self._previous = ParameterContext._current
        ParameterContext._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        """
        ParameterContext._current = self._previous

    @classmethod
    def get_current(cls):
        """
        Get the current context manager.
        """
        return cls._current


#
### VALUE TRAINABLE PARAMETERS. ###
#


#
class ValueTrainableParameter(Value):
    """A specialized `Value` that represents a learnable parameter.

    ValueTrainableParameter encapsulates a scalar value that can be optimized
    using either PyTorch's Autograd (when using `getitem_torch`) or the engine's
    manual differentiation (when using `getitem_np` and `backward`).
    """

    @property
    def value(self) -> Any:
        """
        Get the value of the parameter.
        """
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        """
        Set the value of the parameter.
        """
        self._value = val

    #
    def __init__(self, initial_value: float | int, name: str | None = None) -> None:
        """
        Initialize the parameter.
        """
        #
        super().__init__()

        self.name = name
        self.initial_value = initial_value
        self._value: Any = None

        # Check for active context
        ctx = ParameterContext.get_current()

        # Default behavior: use torch if available
        use_torch_local = HAS_TORCH

        # Initial value setup
        if ctx:
            if ctx.capture:
                # Training mode
                ctx.captured_params.append(self)
            elif ctx.parameters:
                # Inference/Injection mode
                injected_value = None
                if name and name in ctx.parameters:
                    injected_value = ctx.parameters[name]
                if injected_value is not None:
                    self._value = float(injected_value)
                    use_torch_local = False

        # If no injected value, use initial
        if self._value is None:
            if use_torch_local:
                self._value = torch.tensor(initial_value, dtype=torch.float32)
            else:
                self._value = float(initial_value)

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """
        Get the value of the parameter.
        """

        #
        val = self.value
        # Check if we need to capture this parameter in the current context
        ctx = ParameterContext.get_current()
        if ctx and ctx.capture:
            if self not in ctx.captured_params:
                ctx.captured_params.append(self)

        if HAS_TORCH and isinstance(val, torch.Tensor):
            return float(val.item())

        #
        try:
            return float(val)
        except (TypeError, ValueError):
            # Fallback for autograd boxes which might not implement __float__ directly
            # in some contexts, but usually they do.
            if hasattr(val, "_value"):
                return float(val._value)
            return val

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """
        Get the value of the parameter.
        """

        #
        val = self.value

        # Check if we need to capture this parameter in the current context
        ctx = ParameterContext.get_current()
        if ctx and ctx.capture:
            if self not in ctx.captured_params:
                ctx.captured_params.append(self)

        # Robust scalar extraction for torch
        if HAS_TORCH and isinstance(val, torch.Tensor):
            val = val.item()

        #
        return np.ones_like(indexes_buffer, dtype=indexes_buffer.dtype) * val

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """
        Get the value of the parameter.
        """

        #
        ### Best way to pass the value correctly with good gradient flow. ###
        #
        # Check if we need to capture this parameter in the current context
        ctx = ParameterContext.get_current()
        if ctx and ctx.capture:
            if self not in ctx.captured_params:
                ctx.captured_params.append(self)

        if isinstance(self.value, float):
            # Fallback if we accidentally call torch render on inference object
            # Convert float to tensor on fly (no gradient obviously)
            t_val = torch.tensor(self.value, device=device)
            return t_val.expand_as(indexes_buffer)

        #
        return self.value.to(device).expand_as(indexes_buffer)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Accumulate gradients for this parameter.
        """
        engine = context.get("engine")
        if engine:
            # Accumulate gradient: dL/dp = sum(dL/dy * dy/dp)
            # For a parameter p, dy/dp = 1 (constant value across indices)
            # So the gradient is just the sum of grad_output.
            # Use float64 for accumulation to avoid precision loss
            grad = np.sum(grad_output, dtype=np.float64)

            # Use the engine's gradient dictionary
            if hasattr(engine, "gradients"):
                if self in engine.gradients:
                    engine.gradients[self] += grad
                else:
                    engine.gradients[self] = np.array([grad], dtype=np.float64)

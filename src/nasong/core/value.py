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
TODO: add full docstring, explaining what the goal of this script is, and explaining for each class and each function what is it, how it works, and how to use it.
"""

#
### Import Modules. ###
#
from typing import Any, Dict

#
import numpy as np
from numpy.typing import NDArray

#
try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = Any  # type: ignore # Mock for imports

    class Tensor:
        pass  # Mock for runtime type hints


#
### ABSTRACT CLASS. ###
#


#
class Value:
    """
    Abstract base class for a time-varying value.

    This class defines the interface for all 'Value' objects, which are used
    to generate signals, envelopes, modulations, etc., on a per-sample basis.
    """

    #
    def __init__(self) -> None:
        """Initializes the base Value object."""

        #
        pass

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """
        Get the value at a single sample index.

        This is the non-vectorized, sample-by-sample method.
        It's often slower and used as a fallback.

        Args:
            index: The sample index (integer).

        Returns:
            The calculated value (float) at that index.
        """

        #
        return 0

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """
        Get the values for an array of sample indexes (vectorized).

        This is the performance-critical method used for rendering audio blocks.

        For the implementation of the base Value class:

            - The base implementation is a slow, non-optimized placeholder
            that iterates and calls get_item.

            - Subclasses should override this
            with a fast, vectorized NumPy implementation.

        Args:
            indexes_buffer: A NumPy array of sample indexes (as floats).

        Returns:
            A NumPy array of calculated values (float32), matching the
            shape of indexes_buffer.
        """

        #
        ### If we arrive here, it is because there are not implemented getitem_np method, so we are using this non optimized placeholder. ###
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
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """
        Get the values for a tensor of sample indexes (vectorized).

        Note: Very very important, the torch part will be used to learn parameters
        for ValueParameter objects.

        So the gradient flow is important and need to be well implemented
        without much discontinuity with all the other torch operations.

        So for instance, we should avoid randint, clamp or using the get_item
        method.

        Args:
            indexes_buffer: A PyTorch tensor of sample indexes (as floats).
            sample_rate: The sample rate.
            device: Device to use for tensor operations ("cpu", "cuda", etc.)

        Returns:
            A PyTorch tensor of calculated values (float32), matching the
            shape of indexes_buffer.
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
        from nasong.core.values.mult_itms_ops.value_sum import Sum
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([self, other])

    def __radd__(self, other):
        from nasong.core.values.mult_itms_ops.value_sum import Sum
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([other, self])

    def __mul__(self, other):
        from nasong.core.values.mult_itms_ops.value_product import Product
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Product([self, other])

    def __rmul__(self, other):
        from nasong.core.values.mult_itms_ops.value_product import Product
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Product([other, self])

    def __sub__(self, other):
        from nasong.core.values.mult_itms_ops.value_sum import Sum
        from nasong.core.values.basic.value_constant import Constant

        # self - other = self + (other * -1)
        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([self, other * Constant(-1.0)])

    def __rsub__(self, other):
        from nasong.core.values.mult_itms_ops.value_sum import Sum
        from nasong.core.values.basic.value_constant import Constant

        # other - self = other + (self * -1)
        if not isinstance(other, Value):
            other = Constant(other)
        return Sum([other, self * Constant(-1.0)])

    def __truediv__(self, other):
        from nasong.core.values.mult_itms_ops.value_product import Product
        from nasong.core.values.basic.value_constant import Constant
        from nasong.core.values.complex.value_pow import Pow

        if not isinstance(other, Value):
            other = Constant(other)
        # self / other = self * (other ** -1)
        return Product([self, Pow(other, Constant(-1.0))])

    def __rtruediv__(self, other):
        from nasong.core.values.mult_itms_ops.value_product import Product
        from nasong.core.values.basic.value_constant import Constant
        from nasong.core.values.complex.value_pow import Pow

        if not isinstance(other, Value):
            other = Constant(other)
        return Product([other, Pow(self, Constant(-1.0))])

    def __mod__(self, other):
        from nasong.core.values.single_itms_ops.value_modulo import Modulo
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Modulo(self, other)

    def __rmod__(self, other):
        from nasong.core.values.single_itms_ops.value_modulo import Modulo
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Modulo(other, self)

    def __pow__(self, other):
        from nasong.core.values.complex.value_pow import Pow
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Pow(self, other)

    def __rpow__(self, other):
        from nasong.core.values.complex.value_pow import Pow
        from nasong.core.values.basic.value_constant import Constant

        if not isinstance(other, Value):
            other = Constant(other)
        return Pow(other, self)

    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Calculates gradients for the NumPy engine (manual differentiation).

        Subclasses should override this to propagate gradients to their inputs
        and update their internal parameters.

        Args:
            grad_output: The gradient of the loss with respect to this node's output.
            context: A storage for intermediate values from the forward pass.
            sample_rate: The sample rate.
        """
        pass


#
### CONTEXT MANAGER FOR TRAINABLE PARAMETERS ###
#


#
class ParameterContext:
    """
    Context manager to capture or inject parameters into ValueTrainableParameter.
    """

    _current = None

    def __init__(
        self,
        parameters: dict[str, float] | None = None,
        capture: bool = False,
        ignore_unknown: bool = True,
    ):
        self.parameters = parameters or {}
        self.capture = capture
        self.captured_params: list["ValueTrainableParameter"] = []
        self.ignore_unknown = ignore_unknown
        self._param_counter = 0

    def __enter__(self):
        self._previous = ParameterContext._current
        ParameterContext._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ParameterContext._current = self._previous

    @classmethod
    def get_current(cls):
        return cls._current


#
### VALUE TRAINABLE PARAMETERS. ###
#


#
class ValueTrainableParameter(Value):
    """
    A Value that can be trained.
    """

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        self._value = val

    #
    def __init__(self, initial_value: float | int, name: str | None = None) -> None:

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
    #
    def get_item(self, index: int, sample_rate: int) -> float:

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

        # For standard numpy, we want to ensure val is a scalar float.
        # For autograd, val will be an ArrayBox.
        # Using ones_like * val is a robust way to broadcast that works for both.
        # IMPORTANT: Autograd + float32 often fails. We use the dtype of the indices (usually float64 in autograd)
        return np.ones_like(indexes_buffer, dtype=indexes_buffer.dtype) * val

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:

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

        return self.value.to(device).expand_as(indexes_buffer)

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: Dict[str, Any],
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

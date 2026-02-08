#
### Import Modules. ###
#
from typing import Any

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
class _FloatWrapper(float):
    """Wraps a float to provide a .item() method for compatibility with Torch-style code."""

    def item(self):
        return self


#
### VALUE TRAINABLE PARAMETERS. ###
#


#
class ValueTrainableParameter(Value):
    """
    A Value that can be trained.
    """

    #
    def __init__(self, initial_value: float | int, name: str | None = None) -> None:

        #
        super().__init__()

        self.name = name
        self.initial_value = initial_value

        # Check for active context
        ctx = ParameterContext.get_current()

        # Default behavior: use torch if available
        use_torch_local = HAS_TORCH

        # Value to hold (Tensor if training/torch, float if inference/no-torch)
        self.value: Any = None

        if ctx:
            if ctx.capture:
                # Training mode restriction: must have torch
                if not HAS_TORCH:
                    pass

                if name is None:
                    # Auto-generate name based on order/counter if needed
                    pass

                ctx.captured_params.append(self)

            elif ctx.parameters:
                # Inference/Injection mode
                injected_value = None

                if name and name in ctx.parameters:
                    injected_value = ctx.parameters[name]
                else:
                    pass

                if injected_value is not None:
                    # We found a value! Use it and force NO-TORCH mode for this instance (inference)
                    self.value = _FloatWrapper(injected_value)
                    use_torch_local = False

        # If no injected value, use initial
        if self.value is None:
            if use_torch_local:
                self.value = torch.tensor(initial_value, dtype=torch.float32)
            else:
                self.value = _FloatWrapper(initial_value)

    #
    def get_item(self, index: int, sample_rate: int) -> float:

        #
        if isinstance(self.value, float):
            return self.value
        return self.value.item()

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        #
        val = self.value
        if not isinstance(val, float):
            val = val.item()

        return np.full_like(indexes_buffer, fill_value=val, dtype=np.float32)

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
        if isinstance(self.value, float):
            # Fallback if we accidentally call torch render on inference object
            # Convert float to tensor on fly (no gradient obviously)
            t_val = torch.tensor(self.value, device=device)
            return t_val.expand_as(indexes_buffer)

        return self.value.to(device).expand_as(indexes_buffer)

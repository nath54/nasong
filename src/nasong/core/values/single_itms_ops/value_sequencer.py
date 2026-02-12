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
Sequencer operation implementation.

This module provides the `Sequencer` class, which manages a collection of
triggered events (notes) and sums their outputs.

Example:
    >>> from nasong.core.values.basic.value_constant import Constant
    >>> from nasong.core.values.single_itms_ops.value_sequencer import Sequencer
    >>> def factory(t, f, s, d): return Constant(f) # Dummy factory
    >>> data = [(440.0, 0, 1), (880.0, 1, 1)]
    >>> seq = Sequencer(Constant(0), factory, data)
    >>> seq.get_item(0, 44100)
    1320.0
"""

#
### Import Modules. ###
#
from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.mult_itms_ops.value_sum import Sum


#
class Sequencer(Value):
    """A Value container that generates a sequence of "notes" or "events".

    This class automates the process of summing multiple `Value` objects
    that are triggered at different times.

    Attributes:
        sum (Sum): The internal Sum object containing all generated notes.
    """

    #
    def __init__(
        self,
        time: Value,
        instrument_factory: Callable[..., Value],
        note_data_list: list[tuple[Any, ...]],
    ) -> None:
        """Initializes the Sequencer.

        Args:
            time (Value): The master time Value to pass to each note.
            instrument_factory (Callable[..., Value]): A function that creates
                a Value object for each data tuple. Expected signature:
                factory(time, *note_data).
            note_data_list (list[tuple[Any, ...]]): A list of tuples containing
                the arguments for each instrument factory call.

        Raises:
            ValueError: If the instrument factory returns None.
        """

        #
        super().__init__()

        #
        ### Build the list of all note/event Value objects. ###
        #
        notes: list[Value] = []
        #
        for note_data in note_data_list:
            #
            # Call the factory, e.g.:
            #   PianoNote(time, *note_data)
            # where note_data = (frequency, start_time, duration)
            #
            val = instrument_factory(time, *note_data)
            if val is None:
                raise ValueError(
                    f"Instrument factory returned None for note_data: {note_data}"
                )
            notes.append(val)

        #
        ### The sequencer's total output is simply the sum of all notes. ###
        #
        self.sum: Sum = Sum(notes)

    #
    def get_item(self, index: int, sample_rate: int) -> float:
        """Returns the summed sequence value for a specific index.

        Args:
            index (int): The sample index.
            sample_rate (int): The audio sample rate.

        Returns:
            float: The sum of all active notes at the given index.
        """

        #
        ### Proxy the call to the internal Sum object. ###
        #
        return self.sum.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Returns a vectorized NumPy array of the summed sequence values.

        Args:
            indexes_buffer (NDArray[np.float32]): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.

        Returns:
            NDArray[np.float32]: Vectorized sequence samples.
        """

        #
        ### Proxy the call to the internal Sum object. ###
        #
        return self.sum.getitem_np(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate
        )

    #
    def getitem_torch(
        self,
        indexes_buffer: Tensor,
        sample_rate: int,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Generates the summed sequence values for training using PyTorch.

        Args:
            indexes_buffer (Tensor): A buffer of sample indexes.
            sample_rate (int): The audio sample rate.
            device (str | torch.device): The device to use for the tensor.

        Returns:
            Tensor: A tensor of sequence samples.
        """

        #
        ### Proxy the call to the internal Sum object. ###
        #
        return self.sum.getitem_torch(
            indexes_buffer=indexes_buffer, sample_rate=sample_rate, device=device
        )

    #
    def backward(
        self,
        grad_output: NDArray[np.float32],
        context: dict[str, Any],
        sample_rate: int,
    ) -> None:
        """Propagates gradients through the sequencer.

        Proxies the call to the internal Sum object.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the output.
            context (dict[str, Any]): The backward context.
            sample_rate (int): The audio sample rate.
        """
        self.sum.backward(grad_output, context, sample_rate)

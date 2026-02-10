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
from typing import Callable, Any, Dict
import numpy as np
from numpy.typing import NDArray

#
from nasong.core.value import Value
from nasong.core.value import torch, Tensor
from nasong.core.values.mult_itms_ops.value_sum import Sum


#
class Sequencer(Value):
    """
    A Value container that generates a sequence of "notes" or "events".

    This class automates the process of summing multiple `Value` objects
    that are triggered at different times.

    It takes a list of data (e.g., a list of (start_time, frequency, duration)
    tuples) and a "factory" function. It calls the factory for each
    item in the data list to create a `Value` object, and then
    creates a single `Sum` of all the created objects.
    """

    #
    def __init__(
        self,
        time: Value,
        # The factory function must accept `time` as its first argument,
        # followed by the unpacked arguments from the data tuple.
        # e.g.: factory(time, freq, start, dur)
        instrument_factory: Callable[..., Value],
        note_data_list: list[tuple[Any, ...]],
    ) -> None:

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

        #
        ### Proxy the call to the internal Sum object. ###
        #
        return self.sum.get_item(index=index, sample_rate=sample_rate)

    #
    def getitem_np(
        self, indexes_buffer: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

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
        context: Dict[str, Any],
        sample_rate: int,
    ) -> None:
        """
        Propagate gradients through sequencer.
        Proxies to the internal Sum object.
        """
        self.sum.backward(grad_output, context, sample_rate)

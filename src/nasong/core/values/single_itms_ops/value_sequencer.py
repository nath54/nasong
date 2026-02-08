#
### Import Modules. ###
#
from typing import Callable, Any

#
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
            notes.append(instrument_factory(time, *note_data))

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

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
Song rendering and export engine.

This module provides the `Song` class, which manages the rendering of
`Value` objects into audio data and handles exporting the result to WAV files.
It supports both NumPy and PyTorch backends.
"""

#
### Import Modules. ###
#
from typing import Callable

#
try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

    class Tensor:
        """
        A placeholder class for PyTorch tensors when PyTorch is not available.
        """

        pass  # pylint: disable=unnecessary-pass

    class torch:  # pylint: disable=invalid-name
        """
        A placeholder class for PyTorch when PyTorch is not available.
        """

        class device:  # pylint: disable=invalid-name
            """
            A placeholder class for PyTorch device when PyTorch is not available.
            """

            def __init__(self, *args):
                pass

        class cuda:  # pylint: disable=invalid-name
            """
            A placeholder class for PyTorch CUDA when PyTorch is not available.
            """

            @staticmethod
            def is_available():
                """
                Returns False when PyTorch is not available.
                """
                return False

        @staticmethod
        def is_available():
            """
            Returns False when PyTorch is not available.
            """
            return False


#
import numpy as np
from numpy.typing import NDArray

#
# from tqdm import tqdm
#
import nasong.core.config as lc
import nasong.core.all_values as lv
import nasong.core.wav as lw


#
def get_device() -> str | torch.device:
    """Returns the best available device for PyTorch operations (CUDA or CPU).

    Returns:
        str | torch.device: The torch device object or string identifier.
    """


#
class Song:
    """Represents a song or audio segment to be rendered.

    A Song combines a configuration (sample rate, duration) with a time-varying
    Value function to produce a renderable audio signal.

    Attributes:
        config (lc.Config): Configuration settings for the song.
        value_of_time (Callable[[lv.Value], lv.Value]): A function that takes
             a time-varying Value (the "clock") and returns the audio Value tree.
    """

    #
    def __init__(
        self, config: lc.Config, value_of_time: Callable[[lv.Value], lv.Value]
    ) -> None:
        """Initializes the Song.

        Args:
            config (lc.Config): Global audio configuration.
            value_of_time (Callable[[lv.Value], lv.Value]): Function defining
                the audio structure based on time.
        """

        #
        self.config: lc.Config = config
        self.value_of_time: Callable[[lv.Value], lv.Value] = value_of_time

    #
    def render(self) -> NDArray[np.float32]:
        """Renders the song to a NumPy array.

        This method uses the NumPy backend for rendering. It is stable and
        suitable for most synthesis tasks.

        Returns:
            NDArray[np.float32]: The rendered audio signal.
        """

        #
        time_val: lv.Value = lv.BasicScaling(
            value=lv.Identity(),
            mult_scale=lv.Constant(1 / self.config.sample_rate),
            sum_scale=lv.Constant(0),
        )

        #
        audio_value: lv.Value = self.value_of_time(time_val)

        #
        tot_samples: int = int(self.config.sample_rate * self.config.total_duration)

        #
        idx_buffer: NDArray[np.float32] = np.arange(0, tot_samples, 1, dtype=np.float32)
        #
        audio_data: NDArray[np.float32] = audio_value.getitem_np(
            indexes_buffer=idx_buffer, sample_rate=self.config.sample_rate
        )

        #
        return audio_data

    #
    def render_torch(self, device: str | torch.device = get_device()) -> Tensor:
        """Renders the song to a PyTorch tensor.

        This method uses the PyTorch backend, enabling hardware acceleration
        and gradient-based optimization of the song's parameters.

        Args:
            device (str | torch.device, optional): The device to use for
                rendering. Defaults to the result of `get_device()`.

        Returns:
            Tensor: The rendered audio signal as a torch tensor.
        """

        #
        time_val: lv.Value = lv.BasicScaling(
            value=lv.Identity(),
            mult_scale=lv.Constant(1 / self.config.sample_rate),
            sum_scale=lv.Constant(0),
        )

        #
        audio_value: lv.Value = self.value_of_time(time_val)

        #
        tot_samples: int = int(self.config.sample_rate * self.config.total_duration)

        #
        idx_buffer: Tensor = torch.arange(
            tot_samples, dtype=torch.float32, device=device
        )
        #
        audio_data: Tensor = audio_value.getitem_torch(
            indexes_buffer=idx_buffer,
            sample_rate=self.config.sample_rate,
            device=device,
        )

        #
        return audio_data

    #
    def export_to_wav(
        self, use_torch: bool = False, device: str | torch.device = get_device()
    ) -> None:
        """Renders the song and saves it as a WAV file.

        The output filename and sample rate are determined by the song's config.

        Args:
            use_torch (bool, optional): Whether to use the PyTorch backend
                for rendering. Defaults to False.
            device (str | torch.device, optional): Device for PyTorch rendering.
                Defaults to the result of `get_device()`.
        """

        #
        audio_data: NDArray[np.float32]

        #
        if use_torch:
            #
            audio_data = self.render_torch(device=device).cpu().numpy()
        #
        else:
            #
            audio_data = self.render()

        #
        prepared_audio_signal: NDArray[np.int16] = lw.WavUtils.prepare_signal(
            audio_data=audio_data
        )

        #
        lw.WavUtils.save_wav_file(
            filename=self.config.output_filename,
            sample_rate=self.config.sample_rate,
            audio_data=prepared_audio_signal,
        )

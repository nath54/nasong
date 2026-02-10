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
import numpy as np
from numpy.typing import NDArray

#
from scipy.io import wavfile


#
class WavUtils:
    #
    @staticmethod
    def prepare_signal(audio_data: NDArray[np.float32]) -> NDArray[np.int16]:

        #
        ### The amplitude is scaled by 32767 for a 16-bit integer representation ###
        ### (which is common for WAV files)                                      ###
        #
        max_amplitude_16bit: float = 32767.0

        #
        ### Convert the floating-point signal to 16-bit integers                 ###
        ### This scaling is necessary because wavfile.write expects integer data ###
        ### for standard PCM audio formats.                                      ###
        #
        return (audio_data * max_amplitude_16bit).astype(np.int16)

    #
    @staticmethod
    def save_wav_file(
        filename: str, sample_rate: int, audio_data: NDArray[np.int16]
    ) -> None:
        """
        Saves a NumPy array to a WAV file.

        Args:
            filename: The path and name for the output WAV file.
            sample_rate: The sample rate (Hz) of the audio data.
            audio_data: The NumPy array containing the audio signal.
        """

        #
        try:
            #
            ### wavfile.write automatically handles the data type (np.int16 in this case). ###
            #
            wavfile.write(filename, sample_rate, audio_data)  # type: ignore
            #
            print(f"[SUCCESS] Successfully saved audio to '{filename}'")
        #
        except Exception as e:
            #
            print(f"[ERROR] Error saving WAV file: {e}")

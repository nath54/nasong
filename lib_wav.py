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
    def prepare_signal(
        audio_data: NDArray[np.float32]
    ) -> NDArray[np.int16]:

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
        audio_data: np.ndarray = (raw_signal * max_amplitude_16bit).astype(np.int16)

    #
    @staticmethod
    def save_wav_file(
        filename: str,
        sample_rate: int,
        audio_data: NDArray[np.int16]
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
            wavfile.write(filename, sample_rate, audio_data)
            #
            print(f"✅ Successfully saved audio to '{filename}'")
        #
        except Exception as e:
            #
            print(f"❌ Error saving WAV file: {e}")




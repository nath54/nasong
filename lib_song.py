#
### Import Modules. ###
#
from typing import Callable
#
import numpy as np
from numpy.typing import NDArray
#
# from tqdm import tqdm
#
import lib_config as lc
import lib_value as lv
import lib_wav as lw


#
class Song:

    #
    def __init__(
        self,
        config: lc.Config,
        value_of_time: Callable[[lv.Value], lv.Value]
    ) -> None:

        #
        self.config: lc.Config = config
        self.value_of_time: Callable[[lv.Value], lv.Value] = value_of_time

    #
    def render_single_thread(
        self,
        from_sample: int,
        to_sample: int,
        audio_value: lv.Value,
        audio_data: NDArray[np.float32]
    ) -> None:

        """
        I had a lot of doubts concerning this function, because I was wondering about rendering the most efficiently possible (by chunking & parallelisation), and also wanted to be able to easily visualize the rendering progress.
        But, in another hand, I also wanted to introduce effects like echo, reverb, delay, or others that need to have the full audio data (and especially the previously rendered audio steps) to calculate the next step.
        """

        #
        idx_buffer: NDArray[np.float32] = np.arange(from_sample, to_sample, 1, dtype=np.float32)
        #
        audio_data[from_sample: to_sample] = audio_value.getitem_np(indexes_buffer=idx_buffer, sample_rate=self.config.sample_rate)

        #
        # steps: int = 2048 * 10

        # #
        # for i in tqdm(range(from_sample, to_sample, steps)):
        #     #
        #     idx_buffer: NDArray[np.float32] = np.arange(i, min(to_sample, i+steps), 1, dtype=np.float32)
        #     #
        #     audio_data[i:min(to_sample, i+steps)] = audio_value.getitem_np(indexes_buffer=idx_buffer, sample_rate=self.config.sample_rate)

    #
    def render(self) -> NDArray[np.float32]:

        #
        time_val: lv.Value = lv.BasicScaling(
            value=lv.Identity(),
            mult_scale=lv.Constant(1/self.config.sample_rate),
            sum_scale=lv.Constant(0)
        )

        #
        audio_value: lv.Value = self.value_of_time(time_val)

        #
        tot_samples: int = int(self.config.sample_rate * self.config.total_duration)

        #
        audio_data: NDArray[np.float32] = np.zeros((tot_samples,), dtype=np.float32)

        #
        self.render_single_thread(
            from_sample=0,
            to_sample=tot_samples,
            audio_value=audio_value,
            audio_data=audio_data
        )

        #
        return audio_data

    #
    def export_to_wav(self) -> None:

        #
        audio_data: NDArray[np.float32] = self.render()

        #
        prepared_audio_signal: NDArray[np.int16] = lw.WavUtils.prepare_signal(audio_data=audio_data)

        #
        lw.WavUtils.save_wav_file(
            filename=self.config.output_filename,
            sample_rate=self.config.sample_rate,
            audio_data=prepared_audio_signal
        )


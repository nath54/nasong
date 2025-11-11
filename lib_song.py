#
### Import Modules. ###
#
from typing import Callable
#
import numpy as np
from numpy.typing import NDArray
#
from tqdm import tqdm
import threading
import concurrent.futures
import multiprocessing
import os
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
        self.value_of_time: lv.Value = value_of_time

    #
    def render_single_thread(
        self,
        from_sample: int,
        to_sample: int,
        audio_value: lv.Value,
        audio_data: NDArray[np.float32]
    ) -> None:


        #
        steps: int = 2048

        #
        for i in tqdm(range(from_sample, to_sample, steps)):
            #
            idx_buffer: NDArray[np.float32] = np.arange(i, min(to_sample, i+steps), 1, dtype=np.float32)
            #
            audio_data[i:min(to_sample, i+steps)] = audio_value.getitem_np(indexes_buffer=idx_buffer)

        #
        # idx_buffer = np.arange(from_sample, to_sample, 1, dtype=np.float32)
        #
        # audio_data[:] = audio_value.getitem_np(indexes_buffer=idx_buffer)

        # #
        # for i in tqdm(range(from_sample, to_sample)):
        #     #
        #     audio_data[i] = audio_value.__getitem__(index=i)



    #
    def render(self, threads: int = -1) -> NDArray[np.float32]:

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


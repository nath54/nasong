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
        for i in tqdm(range(from_sample, to_sample)):
            #
            audio_data[i] = audio_value.__getitem__(index=i)

    #
    def render_multi_thread(
        self,
        tot_samples: int,
        audio_value: lv.Value,
        audio_data: NDArray[np.float32],
        threads: int = multiprocessing.cpu_count()
    ) -> None:

        #
        ### Split the work into chunks for each thread. ###
        #
        chunk_size: int = tot_samples // threads
        chunks: list[tuple[int, int]] = []

        #
        for i in range(threads):
            #
            start_idx: int = i * chunk_size
            end_idx: int = start_idx + chunk_size if i < threads - 1 else tot_samples
            chunks.append((start_idx, end_idx, i))

        #
        ### Create a master progress bar for overall progress. ###
        #
        with tqdm(total=tot_samples, desc="Overall Progress", position=0, leave=True, unit="samples") as master_pbar:

            #
            ### Use ThreadPoolExecutor for parallel processing. ###
            #
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

                #
                ### Submit tasks to the thread pool. ###
                #
                futures = []

                #
                ### For thread-safe progress updates. ###
                #
                lock = threading.Lock()

                #
                def worker(chunk_start: int, chunk_end: int, thread_id: int):

                    """Worker function that processes a chunk and updates progress"""

                    #
                    local_progress: int = 0

                    #
                    ### Process the chunk. ###
                    #
                    for i in range(chunk_start, chunk_end):

                        #
                        audio_data[i] = audio_value.__getitem__(index=i)

                        #
                        local_progress += 1

                        #
                        ### Update master progress bar periodically to avoid overhead. ###
                        ### Update every 1000 samples. ###
                        #
                        if local_progress % 1000 == 0:
                            #
                            with lock:
                                #
                                master_pbar.update(1000)

                    #
                    ### Final update for remaining samples. ###
                    #
                    remaining = (chunk_end - chunk_start) % 1000
                    #
                    if remaining > 0:
                        #
                        with lock:
                            #
                            master_pbar.update(remaining)

                    #
                    return chunk_end - chunk_start

                #
                ### Submit all chunks. ###
                #
                for start_idx, end_idx, thread_id in chunks:
                    #
                    future = executor.submit(worker, start_idx, end_idx, thread_id)
                    futures.append(future)

                #
                ### Wait for all threads to complete. ###
                #
                concurrent.futures.wait(futures)


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
        if threads <= 1:
            #
            self.render_single_thread(
                from_sample=0,
                to_sample=tot_samples,
                audio_value=audio_value,
                audio_data=audio_data
            )

        #
        else:
            #
            self.render_multi_thread(
                tot_samples=tot_samples,
                audio_value=audio_value,
                audio_data=audio_data,
                threads=threads
            )

        #
        return audio_data

    #
    def export_to_wav(self) -> None:

        #
        audio_data: NDArray[np.float32] = self.render()

        #
        lw.WavUtils.save_wav_file(
            filename=self.config.output_filename,
            sample_rate=self.config.sample_rate,
            audio_data=audio_data
        )


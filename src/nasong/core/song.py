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
        pass

    class torch:
        class device:
            def __init__(self, *args):
                pass

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def is_available():
            return False


#
import numpy as np
from numpy.typing import NDArray

#
# from tqdm import tqdm
#
import nasong.core.config as lc
import nasong.core.value as lv
import nasong.core.wav as lw


#
def get_device() -> str | torch.device:

    #
    if torch.cuda.is_available():
        #
        return torch.device("cuda")
    #
    else:
        #
        return torch.device("cpu")


#
class Song:
    #
    def __init__(
        self, config: lc.Config, value_of_time: Callable[[lv.Value], lv.Value]
    ) -> None:

        #
        self.config: lc.Config = config
        self.value_of_time: Callable[[lv.Value], lv.Value] = value_of_time

    #
    def render(self) -> NDArray[np.float32]:

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

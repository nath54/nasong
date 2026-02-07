#
### Import Modules. ###
#
from typing import Callable

#
try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

    class torch:
        class device:
            def __init__(self, *args):
                pass

        def is_available():
            return False


#
import argparse

#
import nasong.core.utils as li
import nasong.core.config as lc
import nasong.core.song as ls
import nasong.core.value as lv


#
def main(
    sound_file: str,
    output_filename: str = "output.wav",
    sample_rate: int = 44100,
    use_torch: bool = False,
    device: str | torch.device = ls.get_device(),
) -> None:
    """
    Main function to orchestrate the sound generation and saving process.
    """

    #
    sound_file_obj: object = li.import_module_from_filepath(filepath=sound_file)

    #
    duration: float = getattr(sound_file_obj, "duration")

    #
    function_of_time: Callable[[lv.Value], lv.Value] = getattr(sound_file_obj, "song")

    #
    song: ls.Song = ls.Song(
        config=lc.Config(
            sample_rate=sample_rate,
            total_duration=duration,
            output_filename=output_filename,
        ),
        value_of_time=function_of_time,
    )

    #
    song.export_to_wav(use_torch=use_torch, device=device)


#
if __name__ == "__main__":
    #
    ### Initialize cli arguments parser ###
    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    #
    ### Specify Arguments. ###
    #
    parser.add_argument(
        "-i", type=str, required=True, help="Path to the python song description."
    )
    parser.add_argument(
        "-o", type=str, default="output.wav", help="Path to the generated file."
    )
    parser.add_argument("-s", type=int, default=44100, help="Sample Rate")
    parser.add_argument(
        "-t", action="store_true", default=False, help="Use torch for rendering."
    )
    parser.add_argument(
        "-d", type=str, default="cpu", help="Device to use for rendering."
    )

    #
    ### Parse Arguments. ###
    #
    args: argparse.Namespace = parser.parse_args()

    #
    ### Call Main. ###
    #
    main(
        sound_file=args.i,
        output_filename=args.o,
        sample_rate=args.s,
        use_torch=args.t,
        device=args.d,
    )

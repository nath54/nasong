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


def run_generation(
    sound_file: str,
    output_filename: str = "output.wav",
    sample_rate: int = 44100,
    use_torch: bool = False,
    device: str | torch.device = "cpu",
) -> None:
    """
    Orchestrate the sound generation and saving process.
    """

    #
    sound_file_obj: object = li.import_module_from_filepath(filepath=sound_file)

    #
    duration: float = getattr(sound_file_obj, "duration")

    #
    function_of_time: Callable[[lv.Value], lv.Value] = getattr(sound_file_obj, "song")

    #
    if use_torch:
        if not HAS_TORCH:
            print("Warning: Torch requested but not available. Falling back to NumPy.")
            use_torch = False
        elif isinstance(device, str):
            device = torch.device(device)

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


def main():
    #
    ### Initialize cli arguments parser ###
    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Nasong: Generate audio from Python song descriptions."
    )

    #
    ### Specify Arguments. ###
    #
    parser.add_argument(
        "input_file", type=str, nargs="?", help="Path to the python song description."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the python song description (alternative).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Path to the generated file.",
    )
    parser.add_argument(
        "-s", "--sample-rate", type=int, default=44100, help="Sample Rate"
    )
    parser.add_argument(
        "-t",
        "--torch",
        action="store_true",
        default=False,
        help="Use torch for rendering.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="Device to use for rendering."
    )

    #
    ### Parse Arguments. ###
    #
    args: argparse.Namespace = parser.parse_args()

    # Handle input file from positional or flag
    input_path = args.input_file or args.input

    if not input_path:
        parser.print_help()
        return

    #
    ### Call Generation Logic. ###
    #
    run_generation(
        sound_file=input_path,
        output_filename=args.output,
        sample_rate=args.sample_rate,
        use_torch=args.torch,
        device=args.device,
    )


#

if __name__ == "__main__":
    main()

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


"""Stand-alone audio generation utility.

This script serves as the primary entry point for rendering audio from NaSong
composition files (.py). It handles command-line arguments for sample rate,
output path, and hardware acceleration settings.
"""

#
### Import Modules. ###
#
from typing import Callable


#
import argparse

#
import nasong.core.utils as li
import nasong.core.config as lc
import nasong.core.song as ls
import nasong.core.all_values as lv


#
HAS_TORCH: bool = True
#
try:
    import torch
#
except (ImportError, OSError):
    #
    HAS_TORCH = False

    #
    ### Mock torch classes ###
    #
    class torch:
        #
        class device:
            def __init__(self, *args):
                pass

        #
        @staticmethod
        def is_available():
            return False


#
def run_generation(
    sound_file: str,
    output_filename: str = "output.wav",
    sample_rate: int = 44100,
    use_torch: bool = False,
    device: str | torch.device = "cpu",
) -> None:
    """Orchestrates the sound generation and saving process.

    Loads a song module, initializes the rendering engine (NumPy or Torch),
    and exports the result to a WAV file.

    Args:
        sound_file (str): Path to the Python song description file.
        output_filename (str, optional): Target WAV file path. Defaults to "output.wav".
        sample_rate (int, optional): Audio sampling rate. Defaults to 44100.
        use_torch (bool, optional): Whether to use PyTorch for rendering.
            Defaults to False.
        device (str | torch.device, optional): Device for Torch rendering.
            Defaults to "cpu".
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


def main() -> None:
    """Main entry point for the audio generation CLI."""
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

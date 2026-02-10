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
import sounddevice as sd


def list_audio_devices():
    devices = sd.query_devices()
    default_out = sd.default.device[1]

    with open("audio_info_v2.txt", "w", encoding="utf-8") as f:
        f.write("\n--- Available Audio Devices ---\n")
        for i, dev in enumerate(devices):
            marker = ">> DEFAULT OUT <<" if i == default_out else ""
            f.write(
                f"[{i}] {dev['name']} (Channels: Out={dev['max_output_channels']}, In={dev['max_input_channels']}) {marker}\n"
            )

        f.write("\n--- Current Selection ---\n")
        if default_out == -1:
            f.write("Selected Output: None (Check system settings!)\n")
        else:
            current = devices[default_out]
            f.write(f"Index: {default_out}\n")
            f.write(f"Name: {current['name']}\n")
            f.write(f"Sample Rate: {current['default_samplerate']}\n")

        f.write(
            "\nNote: System-level volume cannot be directly retrieved via sounddevice.\n"
        )
        f.write("Please check your OS Volume Mixer.\n")


if __name__ == "__main__":
    list_audio_devices()

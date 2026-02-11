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
Headless entry point for NaSong live-coding.

This module allows running NaSong scripts without a GUI/TUI, monitoring the
script file for changes and hot-reloading the audio session automatically.
"""

#
### Import Modules. ###
#
import os
import sys
import time
import argparse

#
from nasong.app.live_session import LiveSession


def main():
    """
    Main loop for the headless live-coding environment.

    Parses CLI arguments, initializes a `LiveSession`, and enters a watch-loop
    that reloads the user script whenever the file is saved.
    """
    parser = argparse.ArgumentParser(description="Headless NaSong Algo-Rave (Dev Tool)")
    parser.add_argument("script", help="Path to the user script (e.g., demo_theory.py)")
    parser.add_argument(
        "--volume", type=float, default=0.8, help="Master volume (0.0+)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Audio device index or name"
    )
    parser.add_argument(
        "--rate", type=int, default=44100, help="Sample rate (try 48000 if silent)"
    )
    args = parser.parse_args()

    # Normalize device if it's a digit
    device = args.device
    if device and device.isdigit():
        device = int(device)

    session = LiveSession(device=device, sample_rate=args.rate)
    session.set_volume(args.volume)

    script_abs_path = os.path.abspath(args.script)

    if not os.path.exists(script_abs_path):
        print(f"Error: Script file not found at {script_abs_path}")
        sys.exit(1)

    print("--- Headless Algo-Rave Started ---")
    print(f"Target Script: {script_abs_path}")
    print(f"Device: {device if device is not None else 'Default'}")
    print(f"Sample Rate: {args.rate} Hz")
    print(f"Volume: {args.volume}")
    print("----------------------------------")

    # Initial load
    success = session.load_script(script_abs_path)
    if not success:
        print("Initial load failed. Listening for file changes to retry...")

    # Start audio thread
    try:
        session.start()
        print("Audio stream started.")
    except Exception as e:
        print(f"Critical Error: Could not start audio stream: {e}")
        sys.exit(1)

    last_mtime = os.path.getmtime(script_abs_path)

    print("\n[LIVE] Monitoring for changes. Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.5)
            if os.path.exists(script_abs_path):
                mtime = os.path.getmtime(script_abs_path)
                if mtime > last_mtime:
                    print(
                        f"\n[RELOAD] Detected change in {os.path.basename(script_abs_path)}. Reloading..."
                    )
                    session.load_script(script_abs_path)
                    last_mtime = mtime
    except KeyboardInterrupt:
        print("\nStopping Headless Rave...")
        session.stop()
        print("Audio stream stopped. Exit.")


if __name__ == "__main__":
    main()

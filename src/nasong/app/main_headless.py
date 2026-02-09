import time
import os
import argparse
import sys
from nasong.app.live_session import LiveSession


def main():
    parser = argparse.ArgumentParser(description="Headless NaSong Algo-Rave (Dev Tool)")
    parser.add_argument("script", help="Path to the user script (e.g., demo_theory.py)")
    parser.add_argument(
        "--volume", type=float, default=0.8, help="Master volume (0.0 - 1.0)"
    )
    args = parser.parse_args()

    session = LiveSession()
    session.set_volume(args.volume)

    script_abs_path = os.path.abspath(args.script)

    if not os.path.exists(script_abs_path):
        print(f"Error: Script file not found at {script_abs_path}")
        sys.exit(1)

    print(f"--- Headless Algo-Rave Started ---")
    print(f"Target Script: {script_abs_path}")
    print(f"BPM: (Managed by script)")
    print(f"Volume: {args.volume}")
    print(f"----------------------------------")

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

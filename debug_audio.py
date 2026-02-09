import sys
import os
import time
import numpy as np
import sounddevice as sd

# Add src to path
sys.path.append(os.path.abspath("src"))

from nasong.app.live_session import LiveSession


def debug_audio():
    print("Initializing LiveSession...")
    session = LiveSession(sample_rate=44100)

    script_path = "nasong_examples/song_drum_beat.py"
    print(f"Loading {script_path}...")
    success = session.load_script(script_path)

    if not success:
        print("Failed to load script.")
        return

    print("Script loaded. Checking RenderEngine...")

    # Check if chunks are being generated
    # The render engine runs in a thread. Let's wait a bit.
    time.sleep(2)

    # Manually request a chunk from RenderEngine
    # The engine prioritizes near cursor (0.0)
    chunk = session.render_engine.get_audio_chunk(0)

    if chunk is None:
        print("ERROR: Chunk 0 is None (not rendered yet).")
    else:
        max_amp = np.max(np.abs(chunk))
        print(f"Chunk 0 rendered. Max amplitude: {max_amp}")
        if max_amp == 0:
            print("WARNING: Chunk is silent (all zeros).")
        else:
            print("SUCCESS: Chunk has audio content.")

    # Check playback stream
    print("Starting playback stream for 2 seconds...")
    session.start()
    time.sleep(2)
    session.stop()
    print("Playback test complete.")


if __name__ == "__main__":
    debug_audio()

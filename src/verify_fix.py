import os
import shutil
import json
import numpy as np
import scipy.io.wavfile as wavfile
from nasong.trainable.config import TrainingConfig
from nasong.trainable.train import train_instrument
from nasong.trainable.inference import load_trained_instrument
from nasong.scripts import evaluate, leaderboard
import sys


# Mock sys.argv for scripts
import sys
from unittest.mock import MagicMock

# Mock soundfile if not available
try:
    import soundfile as sf
except ImportError:
    sf = MagicMock()
    sf.read.return_value = (np.zeros(100, dtype=np.float32), 44100)
    sys.modules["soundfile"] = sf

# Also mock matplotlib.pyplot to avoid display issues
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = MagicMock()
    sys.modules["matplotlib.pyplot"] = plt
    sys.modules["matplotlib"] = MagicMock()


class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def verify():
    print("=== STARTING VERIFICATION ===")

    # 1. Setup Dummy Config
    wav_path = "dummy_test.wav"
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wavfile.write(wav_path, sr, (audio * 32767).astype(np.int16))

    test_output_dir = "verification_output"
    if os.path.exists(test_output_dir):
        try:
            shutil.rmtree(test_output_dir)
        except OSError:
            print(
                "Warning: Could not remove existing verification_output directory completely."
            )

    config = TrainingConfig(
        instrument_name="sine",  # Use simple instrument
        target_wav=wav_path,
        epochs=1,
        learning_rate=0.01,
        output_dir=test_output_dir,
        device="cpu",
        engine_type="numpy",  # Test numpy engine fix
    )
    # Short duration
    config.train_duration = 0.1
    config.val_duration = 0.0
    config.test_duration = 0.0

    # 2. Run Training
    print("Running training...")
    try:
        train_instrument(config)
    except Exception as e:
        print(f"FAILED: Training crased with error: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. Check Directory Structure
    print("Check directory structure...")
    subdirs = [
        d
        for d in os.listdir(test_output_dir)
        if os.path.isdir(os.path.join(test_output_dir, d))
    ]
    if not subdirs:
        print("FAILED: No timestamped subdirectory created.")
        return

    timestamp_dir = os.path.join(test_output_dir, subdirs[0])
    print(f"Found timestamp directory: {timestamp_dir}")

    # Check format (roughly)
    import re

    if not re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", subdirs[0]):
        print(
            f"WARNING: Subdirectory {subdirs[0]} does not match expected timestamp format."
        )

    # 4. Check Params Capture
    params_file = os.path.join(timestamp_dir, "params.json")
    if not os.path.exists(params_file):
        print(f"FAILED: params.json not found at {params_file}")
        return

    with open(params_file, "r") as f:
        params = json.load(f)

    if not params:
        print("FAILED: params.json is empty! NumpyEngine parameter capture failed.")
    else:
        print(f"SUCCESS: params.json contains {len(params)} parameters.")

    # 5. Check Leaderboard Discovery (Real execution)
    print("Checking leaderboard discovery (Real)...")
    leaderboard_out = "verification_leaderboard.md"

    # Mock sys.argv
    sys.argv = [
        "nasong-leaderboard",
        "--models-dir",
        test_output_dir,
        "--output",
        leaderboard_out,
    ]
    try:
        leaderboard.main()
        if os.path.exists(leaderboard_out):
            print(f"SUCCESS: Leaderboard generated at {leaderboard_out}")
            with open(
                leaderboard_out, "r", encoding="utf-8"
            ) as f:  # Verify it's readable as utf-8 and contains info
                content = f.read()
                if "sine" in content:
                    print("SUCCESS: Leaderboard content validated.")
                else:
                    print("FAILED: Leaderboard missing experiment info.")
        else:
            print("FAILED: Leaderboard file not created.")
    except Exception as e:
        print(f"FAILED: Leaderboard script execution failed: {e}")
        import traceback

        traceback.print_exc()

    # 6. Check Compatibility with load_trained_instrument
    print("Checking load_trained_instrument compatibility...")
    try:
        instrument_func = load_trained_instrument(timestamp_dir)
        print("SUCCESS: load_trained_instrument returned a function.")
    except Exception as e:
        print(f"FAILED: load_trained_instrument raised exception: {e}")
        import traceback

        traceback.print_exc()

    # 7. Check Evaluate Script Discovery
    print("Checking evaluate script discovery...")
    # Mock sys.argv
    # We use a dummy method or minimal to avoid long execution, or just check discovery logic
    # The script prints "Found X experiments". We can capture stdout or just run it and expect no error/found msg.
    sys.argv = [
        "nasong-evaluate",
        "--models-dir",
        test_output_dir,
        "--methods",
        "legacy",
    ]
    # Note: legacy might be fast enough.

    try:
        # We need to capture stdout to verify "Found 1 experiments"
        from io import StringIO
        import sys as sys_orig

        captured_out = StringIO()
        sys.stdout = captured_out

        evaluate.main()

        sys.stdout = sys_orig.stdout
        output = captured_out.getvalue()
        print("Evaluate Output snippet:", output[:200])

        if "Found 1 experiments" in output:
            print("SUCCESS: Evaluate script found the experiment.")
        else:
            print("FAILED: Evaluate script did not find the experiment.")

    except Exception as e:
        sys.stdout = sys_orig.stdout
        print(f"FAILED: Evaluate script crashed: {e}")
        import traceback

        traceback.print_exc()

    print("=== VERIFICATION COMPLETE ===")


if __name__ == "__main__":
    verify()

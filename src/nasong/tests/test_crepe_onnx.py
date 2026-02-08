import numpy as np

from nasong.trainable.note_detection.onnx_crepe_detect import OnnxCrepeDetector
from nasong.trainable.config import NoteDetectionConfig


def test_onnx_crepe():
    print("Testing OnnxCrepeDetector...")

    # 1. Generate 440Hz Sine Wave
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    audio = 0.5 * np.sin(2 * np.pi * freq * t)

    # Add some silence at the beginning to test timing
    silence = np.zeros(int(0.5 * sr))
    audio = np.concatenate([silence, audio])

    # 2. Initialize Detector
    config = NoteDetectionConfig(method="crepe_onnx", crepe_confidence_threshold=0.6)
    detector = OnnxCrepeDetector(config.__dict__)

    # 3. Detect
    print("Running detection (this may download the model on first run)...")
    try:
        notes = detector.detect(audio, sr)
    except Exception as e:
        print(f"[FAIL] Detection raised exception: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Analyze Results
    print(f"Detected {len(notes)} notes.")
    for i, note in enumerate(notes):
        print(f"Note {i}: {note}")

    if len(notes) == 0:
        print("[FAIL] No notes detected.")
        return

    # Check first note
    note = notes[0]

    # Time check (should start approx 0.5s)
    if abs(note["start_time"] - 0.5) < 0.1:
        print("[OK] Start time correct")
    else:
        print(f"[FAIL] Start time {note['start_time']} != expected 0.5")

    # Freq check
    detected_freq = note["frequencies"][0]
    if abs(detected_freq - 440.0) < 10.0:
        print(f"[OK] Frequency {detected_freq:.2f}Hz is close to 440Hz")
    else:
        print(f"[FAIL] Frequency {detected_freq:.2f}Hz is not close to 440Hz")

    # Amplitude check
    if "amplitude" in note:
        print(f"[OK] Amplitude found: {note['amplitude']:.4f}")
    else:
        print("[FAIL] Amplitude missing")


if __name__ == "__main__":
    test_onnx_crepe()

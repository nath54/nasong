import numpy as np
import inspect
from nasong.trainable.instruments import melodic, bass, atmospheric, synth
from nasong.trainable.note_detection.legacy import LegacyDetector


def verify_instruments():
    instruments = [
        melodic.TrainablePlucked,
        melodic.TrainablePiano,
        melodic.TrainableBowed,
        bass.TrainableBass,
        atmospheric.TrainablePad,
        synth.TrainableSawtoothSynth,
        synth.TrainableSquareSynth,
        synth.TrainableSineSynth,
    ]

    print("Verifying Instrument Signatures...")
    for inst in instruments:
        sig = inspect.signature(inst)
        params = sig.parameters
        if "init_amplitude" in params and "name_prefix" in params:
            print(f"[OK] {inst.__name__} accepts init_amplitude and name_prefix")
        else:
            print(
                f"[FAIL] {inst.__name__} missing arguments. Params: {list(params.keys())}"
            )


def verify_detector():
    print("\nVerifying Legacy Detector Amplitude Extraction...")
    # Create dummy audio (sine wave)
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave with amplitude 0.5
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    detector = LegacyDetector({"legacy_num_notes": 1})
    notes = detector.detect(audio, sr)

    if not notes:
        print("[FAIL] No notes detected")
        return

    note = notes[0]
    print(f"Detected Note: {note}")

    if "amplitude" in note:
        # RMS of sine wave with amp A is A / sqrt(2) = 0.5 * 0.707 = 0.3535
        expected_rms = 0.5 / np.sqrt(2)
        print(
            f"Amplitude: {note['amplitude']:.4f} (Expected RMS approx {expected_rms:.4f})"
        )
        if abs(note["amplitude"] - expected_rms) < 0.05:
            print("[OK] Amplitude calculation is correct")
        else:
            print("[FAIL] Amplitude calculation seems off")
    else:
        print("[FAIL] 'amplitude' key missing in note")


if __name__ == "__main__":
    verify_instruments()
    verify_detector()

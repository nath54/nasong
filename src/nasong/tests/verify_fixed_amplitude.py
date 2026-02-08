import nasong.core.all_values as lv
from nasong.trainable.instruments import melodic, bass, atmospheric, synth


def verify_amplitude_fixed():
    instruments = [
        ("melodic.TrainablePlucked", melodic.TrainablePlucked),
        ("melodic.TrainablePiano", melodic.TrainablePiano),
        ("melodic.TrainableBowed", melodic.TrainableBowed),
        ("bass.TrainableBass", bass.TrainableBass),
        ("atmospheric.TrainablePad", atmospheric.TrainablePad),
        ("synth.TrainableSawtoothSynth", synth.TrainableSawtoothSynth),
        ("synth.TrainableSquareSynth", synth.TrainableSquareSynth),
        ("synth.TrainableSineSynth", synth.TrainableSineSynth),
    ]

    print("Verifying if 'amplitude' is trainable...")

    # Dummy inputs
    time = lv.Value()  # Abstract value
    freq = lv.Constant(440.0)
    start = 0.0
    dur = 1.0

    for name, inst_func in instruments:
        print(f"\nChecking {name}...")

        # Capture parameters
        with lv.ParameterContext(capture=True) as ctx:
            try:
                inst_func(time, freq, start, dur, init_amplitude=0.5)
            except Exception as e:
                # Some instruments might fail with abstract Time, but we just want to see param creation
                # If they do minimal computation before crash, maybe we captured params.
                # Ideally we should pass a valid Time object if needed, but let's try.
                print(f"  (Instantiation warning: {e})")

        # Check captured params
        found_amp = False
        captured_names = []
        for p in ctx.captured_params:
            if p.name:
                captured_names.append(p.name)
                if "_amp" in p.name:
                    found_amp = True

        if found_amp:
            print(
                f"[FAIL] Found trainable amplitude parameter: {[n for n in captured_names if '_amp' in n]}"
            )
        else:
            print(
                f"[OK] No trainable amplitude parameter found. Params: {captured_names}"
            )


if __name__ == "__main__":
    verify_amplitude_fixed()

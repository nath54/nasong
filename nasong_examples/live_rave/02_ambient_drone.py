from nasong.core.all_values import Constant, Sin, LFO, Identity

# === Ambient Drone (Generative) ===
# No sequencer, just pure math functions of Time.

# Global time input
time = Identity()


def drone_voice(freq_base, mod_rate):
    # Slowly drifting frequency
    freq = Constant(freq_base) + LFO(
        time, rate_hz=Constant(mod_rate), waveform_class=Sin, amplitude=Constant(2.0)
    )

    # Slowly drifting amplitude
    amp = LFO(
        time,
        rate_hz=Constant(mod_rate * 0.7),
        waveform_class=Sin,
        amplitude=Constant(0.3),
    ) + Constant(0.5)

    return Sin(time, freq) * amp


# Combine multiple detuned voices
voice1 = drone_voice(110.0, 0.1)  # A2
voice2 = drone_voice(110.5, 0.13)  # A2 detuned
voice3 = drone_voice(164.8, 0.09)  # E3 (Perfect 5th)
voice4 = drone_voice(220.0, 0.05)  # A3 (Octave)

# Sum and scale
sequencer = (voice1 + voice2 + voice3 + voice4) * Constant(0.2)

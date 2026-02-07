import numpy as np
import lib_vis as lv
import lib_wav as lw
import os


def test_vis_tools():
    print("🧪 Testing Visualization Tools...")

    # 1. Generate a simple test signal (Sine wave at 440Hz)
    sr = 44100
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    freq = 440.0
    amplitude = 0.5
    audio_data = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    print(
        f"   Generated sine wave: {len(audio_data)} samples, {freq}Hz, amp={amplitude}"
    )

    # 2. Test Analysis
    analyzer = lv.AudioAnalyzer(audio_data, sr)

    rms = analyzer.get_rms()
    expected_rms = amplitude / np.sqrt(2)
    print(f"   RMS: {rms:.4f} (Expected: ~{expected_rms:.4f})")
    assert np.isclose(rms, expected_rms, rtol=1e-2), "RMS calculation failed"

    peak = analyzer.get_peak_amplitude()
    print(f"   Peak: {peak:.4f} (Expected: {amplitude:.4f})")
    assert np.isclose(peak, amplitude, rtol=1e-2), "Peak calculation failed"

    centroid = analyzer.get_spectral_centroid()
    print(f"   Spectral Centroid: {centroid:.2f} Hz (Expected: ~{freq:.2f} Hz)")
    # Centroid for a pure sine wave should be close to the frequency
    assert np.isclose(centroid, freq, rtol=0.1), "Spectral Centroid failed"

    # Expanded Metrics Tests
    crest, dyn_db = analyzer.get_dynamic_range()
    print(f"   Crest Factor: {crest:.2f}, Dynamic Range: {dyn_db:.2f} dB")
    # Sine wave crest factor is sqrt(2) ~= 1.414
    assert np.isclose(crest, np.sqrt(2), rtol=0.01), "Crest Factor failed"

    flatness = analyzer.get_spectral_flatness()
    print(f"   Spectral Flatness: {flatness:.4f} (Expected: ~0.0 for sine)")
    assert flatness < 0.1, "Spectral Flatness failed (should be low for sine)"

    zcr = analyzer.get_zero_crossing_rate()
    expected_zcr = 2 * freq / sr
    print(f"   Zero Crossing Rate: {zcr:.4f} (Expected: ~{expected_zcr:.4f})")
    assert np.isclose(zcr, expected_zcr, rtol=0.1), "Zero Crossing Rate failed"

    clipping = analyzer.detect_clipping()
    print(f"   Clipping: {clipping:.2f}% (Expected: 0.0%)")
    assert clipping == 0.0, "Clipping detection failed"

    print("✅ Analysis tests passed!")

    # 3. Test Plotting (Check if files are created)
    print("   Testing Plotting functions...")
    lv.plot_waveform(audio_data, sr, filename="test_waveform.png", show=False)
    lv.plot_spectrogram(audio_data, sr, filename="test_spectrogram.png", show=False)
    lv.plot_spectrum(audio_data, sr, filename="test_spectrum.png", show=False)

    assert os.path.exists("test_waveform.png"), "Waveform plot not created"
    assert os.path.exists("test_spectrogram.png"), "Spectrogram plot not created"
    assert os.path.exists("test_spectrum.png"), "Spectrum plot not created"

    print("✅ Plotting tests passed!")

    # Cleanup
    os.remove("test_waveform.png")
    os.remove("test_spectrogram.png")
    os.remove("test_spectrum.png")
    print("🧹 Cleanup done.")


if __name__ == "__main__":
    test_vis_tools()

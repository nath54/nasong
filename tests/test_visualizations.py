import numpy as np
import pytest
import nasong.core.vis as lv


@pytest.fixture
def sine_wave_data():
    sr = 44100
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    freq = 440.0
    amplitude = 0.5
    audio_data = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return audio_data, sr, freq, amplitude


def test_analysis_metrics(sine_wave_data):
    audio_data, sr, freq, amplitude = sine_wave_data
    analyzer = lv.AudioAnalyzer(audio_data, sr)

    # RMS
    rms = analyzer.get_rms()
    expected_rms = amplitude / np.sqrt(2)
    assert np.isclose(rms, expected_rms, rtol=1e-2), "RMS calculation failed"

    # Peak
    peak = analyzer.get_peak_amplitude()
    assert np.isclose(peak, amplitude, rtol=1e-2), "Peak calculation failed"

    # Spectral Centroid
    centroid = analyzer.get_spectral_centroid()
    # Centroid for a pure sine wave should be close to the frequency
    assert np.isclose(centroid, freq, rtol=0.1), "Spectral Centroid failed"

    # Dynamic Range
    crest, dyn_db = analyzer.get_dynamic_range()
    assert np.isclose(crest, np.sqrt(2), rtol=0.01), "Crest Factor failed"

    # Spectral Flatness
    flatness = analyzer.get_spectral_flatness()
    assert flatness < 0.1, "Spectral Flatness failed (should be low for sine)"

    # Zero Crossing Rate
    zcr = analyzer.get_zero_crossing_rate()
    expected_zcr = 2 * freq / sr
    assert np.isclose(zcr, expected_zcr, rtol=0.1), "Zero Crossing Rate failed"

    # Clipping
    clipping = analyzer.detect_clipping()
    assert clipping == 0.0, "Clipping detection failed"


def test_plotting(sine_wave_data, tmp_path):
    audio_data, sr, _, _ = sine_wave_data

    # Use tmp_path fixture for file creation to ensure cleanup
    wave_file = tmp_path / "test_waveform.png"
    spec_file = tmp_path / "test_spectrogram.png"
    spectrum_file = tmp_path / "test_spectrum.png"

    lv.plot_waveform(audio_data, sr, filename=str(wave_file), show=False)
    lv.plot_spectrogram(audio_data, sr, filename=str(spec_file), show=False)
    lv.plot_spectrum(audio_data, sr, filename=str(spectrum_file), show=False)

    assert wave_file.exists(), "Waveform plot not created"
    assert spec_file.exists(), "Spectrogram plot not created"
    assert spectrum_file.exists(), "Spectrum plot not created"

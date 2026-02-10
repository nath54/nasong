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
Visualization and Audio Analysis Module.

This module provides tools for visualizing audio data (waveforms, spectrograms, spectra)
and a comprehensive `AudioAnalyzer` class for extracting various temporal, spectral,
and harmonic features from audio signals. These features are useful for audio verification,
quality assessment, and characteristic analysis of generated or recorded sounds.

Functions:
----------

1.  **`plot_waveform(audio_data, sample_rate, title="Waveform", filename=None, show=True)`**
    *   **Goal:** Visualizes the amplitude of the audio signal over time.
    *   **Usage:** Call with audio data array and sample rate. Optionally save to `filename` or suppress display with `show=False`.

2.  **`plot_spectrogram(audio_data, sample_rate, title="Spectrogram", filename=None, show=True)`**
    *   **Goal:** Visualizes the frequency content of the audio signal over time (time-frequency representation).
    *   **Usage:** Useful for seeing how pitch or timbre evolves. Same usage pattern as `plot_waveform`.

3.  **`plot_spectrum(audio_data, sample_rate, title="Frequency Spectrum", filename=None, show=True)`**
    *   **Goal:** Visualizes the magnitude of frequency components (FFT) for the entire audio clip.
    *   **Usage:** Useful for analyzing the overall tonal balance or identifying dominant frequencies.

Classes:
--------

1.  **`AudioAnalyzer`**
    *   **Goal:** A comprehensive utility for calculating statistical, temporal, and spectral features of an audio signal.
    *   **Usage:** Instantiate with `analyzer = AudioAnalyzer(audio_data, sample_rate)`. Then call specific methods to get metrics.

    **Methods:**

    *   **`get_rms()`**: Returns Root Mean Square amplitude (loudness/power).
    *   **`get_peak_amplitude()`**: Returns the maximum absolute amplitude value.
    *   **`get_spectral_centroid()`**: Returns the "center of mass" of the spectrum (brightness) in Hz.
    *   **`get_envelope(window_size_ms=20.0)`**: Returns the amplitude envelope (moving RMS).
    *   **`compare_similarity(other_audio_data)`**: Computes similarity (0.0-1.0) with another audio signal using cross-correlation.
    *   **`get_dynamic_range()`**: Returns (Crest Factor, Dynamic Range dB).
    *   **`get_temporal_centroid()`**: Returns the center of gravity of energy in time (percussive vs sustained).
    *   **`get_zero_crossing_rate()`**: Returns the rate of sign changes (noisiness/high-freq context).
    *   **`get_spectral_bandwidth()`**: Returns the width of the spectrum around the centroid.
    *   **`get_spectral_flatness()`**: Returns Wiener entropy (0.0=tone, 1.0=noise).
    *   **`detect_clipping(threshold=0.99)`**: Returns percentage of samples exceeding threshold.
    *   **`detect_dc_offset()`**: Returns mean amplitude (should be near 0.0).
    *   **`detect_silence(threshold_db=-60.0)`**: Returns percentage of silent samples.
    *   **`get_adsr_estimates()`**: estimates Attack, Decay, Sustain, Release parameters.
    *   **`get_effective_duration(threshold_db=-40.0)`**: Returns duration of signal above threshold.
    *   **`get_temporal_skewness()`**: Asymmetry of energy envelope.
    *   **`get_temporal_kurtosis()`**: Peakedness of energy envelope.
    *   **`get_spectral_rolloff(percent=0.85)`**: Frequency below which 85% of energy lies.
    *   **`get_spectral_skewness()`**: Asymmetry of the spectrum.
    *   **`get_spectral_kurtosis()`**: Peakedness of the spectrum.
    *   **`get_spectral_slope()`**: Linear regression slope of the magnitude spectrum.
    *   **`get_spectral_flux()`**: Rate of change of the spectrum over time.
    *   **`get_fundamental_frequency()`**: Estimates F0 using autocorrelation.
    *   **`get_harmonic_to_noise_ratio()`**: Estimates HNR in dB.
    *   **`get_odd_even_harmonic_ratio()`**: Ratio of odd vs even harmonic energy.
    *   **`get_tristimulus()`**: Returns (T1, T2, T3) describing harmonic energy distribution.
    *   **`get_pitch_clarity()`**: Normalized autocorrelation peak height (pitch confidence).
"""

#
### Import Modules. ###
#
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram

#
### VISUALIZATION FUNCTIONS. ###
#


def plot_waveform(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    title: str = "Waveform",
    filename: str | None = None,
    show: bool = True,
) -> None:
    """
    Plots the amplitude of the audio signal over time.
    """

    #
    time_axis = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))

    #
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, audio_data)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    #
    if filename:
        plt.savefig(filename)
        print(f"✅ Saved waveform plot to '{filename}'")

    #
    if show:
        plt.show()
    else:
        plt.close()


def plot_spectrogram(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    title: str = "Spectrogram",
    filename: str | None = None,
    show: bool = True,
) -> None:
    """
    Plots the frequency content of the audio signal over time.
    """

    #
    f, t, Sxx = spectrogram(audio_data, sample_rate)

    #
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
    plt.title(title)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Intensity [dB]")
    plt.tight_layout()

    #
    if filename:
        plt.savefig(filename)
        print(f"✅ Saved spectrogram plot to '{filename}'")

    #
    if show:
        plt.show()
    else:
        plt.close()


def plot_spectrum(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    title: str = "Frequency Spectrum",
    filename: str | None = None,
    show: bool = True,
) -> None:
    """
    Plots the magnitude spectrum (FFT) of the audio signal.
    """

    #
    N = len(audio_data)
    yf = fft(audio_data)
    xf = fftfreq(N, 1 / sample_rate)

    #
    ### Only plot the positive frequencies ###
    #
    idx_max = N // 2

    #
    plt.figure(figsize=(10, 4))
    plt.plot(xf[:idx_max], 2.0 / N * np.abs(yf[0:idx_max]))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    #
    if filename:
        plt.savefig(filename)
        print(f"✅ Saved spectrum plot to '{filename}'")

    #
    if show:
        plt.show()
    else:
        plt.close()


#
### ANALYSIS CLASS. ###
#


class AudioAnalyzer:
    """
    Class for analyzing audio data and extracting features for verification.
    """

    #
    def __init__(self, audio_data: NDArray[np.float32], sample_rate: int) -> None:
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    #
    def get_rms(self) -> float:
        """Returns the Root Mean Square (RMS) amplitude."""
        #
        return float(np.sqrt(np.mean(self.audio_data**2)))

    #
    def get_peak_amplitude(self) -> float:
        """Returns the maximum absolute amplitude."""
        #
        return float(np.max(np.abs(self.audio_data)))

    #
    def get_spectral_centroid(self) -> float:
        """
        Returns the spectral centroid (brightness) in Hz.
        """

        #
        magnitudes = np.abs(np.fft.rfft(self.audio_data))
        length = len(self.audio_data)
        freqs = np.fft.rfftfreq(length, d=1.0 / self.sample_rate)

        #
        ### Avoid division by zero ###
        #
        if np.sum(magnitudes) == 0:
            return 0.0

        return float(np.sum(freqs * magnitudes) / np.sum(magnitudes))

    #
    def get_envelope(self, window_size_ms: float = 20.0) -> NDArray[np.float32]:
        """
        Returns the amplitude envelope using a moving RMS window.
        """
        window_samples = int(window_size_ms * self.sample_rate / 1000)
        if window_samples < 1:
            window_samples = 1

        # Simple moving average of squared values, then sqrt
        squared = self.audio_data**2
        window = np.ones(window_samples) / window_samples
        moving_avg = np.convolve(squared, window, mode="same")
        return np.sqrt(moving_avg)

    #
    def compare_similarity(self, other_audio_data: NDArray[np.float32]) -> float:
        """
        Compares this audio with another audio buffer.
        Returns a similarity score (0.0 to 1.0) based on normalized cross-correlation.
        Assumes both signals are aligned and of similar length.
        """

        #
        ### Truncate to the shorter length ###
        #
        min_len = min(len(self.audio_data), len(other_audio_data))
        a = self.audio_data[:min_len]
        b = other_audio_data[:min_len]

        #
        ### Normalize ###
        #
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        #
        ### Avoid division by zero ###
        #
        if norm_a == 0 or norm_b == 0:
            return 0.0 if norm_a == norm_b else 0.0

        #
        ### Calculate dot product ###
        #
        dot_product = np.dot(a, b)
        return float(dot_product / (norm_a * norm_b))

    #
    ### EXPANDED ANALYSIS METRICS ###
    #

    def get_dynamic_range(self) -> tuple[float, float]:
        """
        Returns (Crest Factor, Dynamic Range dB).
        Crest Factor = Peak / RMS.
        Dynamic Range dB = 20 * log10(Peak / RMS).
        """

        #
        peak = self.get_peak_amplitude()
        rms = self.get_rms()

        #
        ### Avoid division by zero ###
        #
        if rms == 0:
            return 0.0, 0.0

        #
        ### Calculate crest factor and dynamic range ###
        #
        crest_factor = peak / rms
        dynamic_range_db = 20 * np.log10(crest_factor)
        #
        return float(crest_factor), float(dynamic_range_db)

    #
    def get_temporal_centroid(self) -> float:
        """
        Returns the temporal centroid (center of gravity in time) in seconds.
        Useful for distinguishing percussive (low centroid) vs sustained (mid centroid) sounds.
        """

        #
        energy = self.audio_data**2
        total_energy = np.sum(energy)

        #
        ### Avoid division by zero ###
        #
        if total_energy == 0:
            return 0.0

        #
        ### Calculate temporal centroid ###
        #
        time_indices = np.arange(len(self.audio_data)) / self.sample_rate
        #
        return float(np.sum(time_indices * energy) / total_energy)

    #
    def get_zero_crossing_rate(self) -> float:
        """
        Returns the Zero Crossing Rate (ZCR).
        High ZCR indicates noisiness or high frequency content.
        """

        #
        zero_crossings = np.nonzero(np.diff(np.signbit(self.audio_data)))[0]
        #
        return float(len(zero_crossings) / len(self.audio_data))

    #
    def get_spectral_bandwidth(self) -> float:
        """
        Returns the spectral bandwidth (width of the spectrum) in Hz.
        """

        #
        centroid = self.get_spectral_centroid()
        magnitudes = np.abs(np.fft.rfft(self.audio_data))
        length = len(self.audio_data)
        freqs = np.fft.rfftfreq(length, d=1.0 / self.sample_rate)

        #
        ### Avoid division by zero ###
        #
        if np.sum(magnitudes) == 0:
            return 0.0

        #
        ### Calculate spectral bandwidth ###
        #
        return float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes))
        )

    #
    def get_spectral_flatness(self) -> float:
        """
        Returns the Spectral Flatness (Wiener Entropy).
        Values near 1.0 indicate noise (white noise), values near 0.0 indicate tonality (sine wave).
        """

        #
        magnitudes = np.abs(np.fft.rfft(self.audio_data))

        #
        ### Add small epsilon to avoid log(0) ###
        #
        magnitudes = magnitudes + 1e-10

        #
        ### Calculate spectral flatness ###
        #
        geometric_mean = np.exp(np.mean(np.log(magnitudes)))
        arithmetic_mean = np.mean(magnitudes)

        #
        return float(geometric_mean / arithmetic_mean)

    #
    def detect_clipping(self, threshold: float = 0.99) -> float:
        """
        Returns the percentage of samples that exceed the threshold (0.0 to 100.0).
        """

        #
        clipped_samples = np.sum(np.abs(self.audio_data) >= threshold)

        #
        return float((clipped_samples / len(self.audio_data)) * 100.0)

    #
    def detect_dc_offset(self) -> float:
        """
        Returns the DC offset (mean amplitude).
        Should be close to 0.0.
        """

        #
        return float(np.mean(self.audio_data))

    #
    def detect_silence(self, threshold_db: float = -60.0) -> float:
        """
        Returns the percentage of the audio that is effectively silent (below threshold_db).
        """

        #
        threshold_amp = 10 ** (threshold_db / 20)
        silent_samples = np.sum(np.abs(self.audio_data) < threshold_amp)
        #
        return float((silent_samples / len(self.audio_data)) * 100.0)

    #
    ### ADVANCED METRICS (INSTRUMENT DEBUGGING) ###
    #

    # --- 1. Time Domain & Envelope ---

    def get_adsr_estimates(self) -> dict[str, float]:
        """
        Estimates ADSR parameters (in seconds) and Sustain Level (0-1).
        This is a heuristic estimation.
        """

        #
        envelope = self.get_envelope(window_size_ms=10)
        max_val = np.max(envelope)

        #
        ### Avoid division by zero ###
        #
        if max_val == 0:
            return {"attack": 0, "decay": 0, "sustain_level": 0, "release": 0}

        #
        ### Normalize ###
        #
        env_norm = envelope / max_val
        peak_idx = np.argmax(env_norm)

        #
        ### Attack: Time to reach peak ###
        #
        attack_time = float(peak_idx / self.sample_rate)

        #
        ### Release: Time from last audible point back to 0 ###
        ### Find last point > 1% ###
        #
        audible_indices = np.where(env_norm > 0.01)[0]
        if len(audible_indices) == 0:
            return {"attack": 0, "decay": 0, "sustain_level": 0, "release": 0}

        last_audible_idx = audible_indices[-1]

        #
        ### Sustain Level estimation: Median of the middle 50% of the note (post-attack, pre-release) ###
        ### If note is too short, just take the value at 50% of duration ###
        #
        start_sustain = peak_idx + int(0.05 * self.sample_rate)  # 50ms after peak
        end_sustain = last_audible_idx - int(0.1 * self.sample_rate)  # 100ms before end

        #
        if start_sustain >= end_sustain:
            sustain_level = env_norm[peak_idx]  # Fallback
            decay_time = 0.0
        else:
            sustain_segment = env_norm[start_sustain:end_sustain]
            sustain_level = float(np.median(sustain_segment))

            #
            ### Decay: Time from peak to reach sustain level ###
            ### Find first point after peak where env drops near sustain level ###
            #
            decay_indices = np.where(
                env_norm[peak_idx:start_sustain] <= sustain_level * 1.1
            )[0]
            #
            if len(decay_indices) > 0:
                decay_time = float(decay_indices[0] / self.sample_rate)
            else:
                decay_time = float((start_sustain - peak_idx) / self.sample_rate)

        #
        ### Release: Time from end of sustain to silence ###
        #
        release_time = float((len(envelope) - last_audible_idx) / self.sample_rate)

        #
        return {
            "attack": attack_time,
            "decay": decay_time,
            "sustain_level": sustain_level,
            "release": release_time,
        }

    #
    def get_effective_duration(self, threshold_db: float = -40.0) -> float:
        """Returns the duration (s) where signal is above threshold."""

        #
        threshold = 10 ** (threshold_db / 20)
        above_thresh = np.where(np.abs(self.audio_data) > threshold)[0]
        #
        if len(above_thresh) == 0:
            return 0.0
        #
        return float((above_thresh[-1] - above_thresh[0]) / self.sample_rate)

    #
    def get_temporal_skewness(self) -> float:
        """Measure of the asymmetry of the energy envelope."""

        #
        env = self.get_envelope()
        centroid = self.get_temporal_centroid()
        t = np.arange(len(env)) / self.sample_rate

        moment3 = np.sum(((t - centroid) ** 3) * env) / np.sum(env)
        moment2 = np.sum(((t - centroid) ** 2) * env) / np.sum(env)

        #
        ### Avoid division by zero ###
        #
        if moment2 == 0:
            return 0.0

        #
        return float(moment3 / (moment2**1.5))

    #
    def get_temporal_kurtosis(self) -> float:
        """Measure of the peakedness of the energy envelope."""

        #
        env = self.get_envelope()
        centroid = self.get_temporal_centroid()
        t = np.arange(len(env)) / self.sample_rate

        #
        moment4 = np.sum(((t - centroid) ** 4) * env) / np.sum(env)
        moment2 = np.sum(((t - centroid) ** 2) * env) / np.sum(env)

        #
        ### Avoid division by zero ###
        #
        if moment2 == 0:
            return 0.0

        #
        ### We remove the excess kurtosis (3.0) to get the final kurtosis ###
        #
        return float(moment4 / (moment2**2)) - 3.0

    # --- 2. Spectral Shape ---

    #
    def _get_mag_spectrum(self) -> tuple[NDArray, NDArray]:
        """Helper to get magnitude spectrum and frequencies."""

        #
        magnitudes = np.abs(np.fft.rfft(self.audio_data))
        freqs = np.fft.rfftfreq(len(self.audio_data), d=1.0 / self.sample_rate)
        #
        return magnitudes, freqs

    #
    def get_spectral_rolloff(self, percent: float = 0.85) -> float:
        """Frequency below which percent% of the total spectral energy lies."""

        #
        mags, freqs = self._get_mag_spectrum()
        energy = mags**2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy)

        idx = np.where(cumulative_energy >= percent * total_energy)[0]
        #
        if len(idx) > 0:
            return float(freqs[idx[0]])
        #
        return 0.0

    #
    def get_spectral_skewness(self) -> float:
        """Asymmetry of the spectrum."""

        #
        mags, freqs = self._get_mag_spectrum()
        centroid = self.get_spectral_centroid()

        #
        sum_mag = np.sum(mags)
        #
        if sum_mag == 0:
            return 0.0

        #
        moment3 = np.sum(((freqs - centroid) ** 3) * mags) / sum_mag
        moment2 = np.sum(((freqs - centroid) ** 2) * mags) / sum_mag

        #
        ### Avoid division by zero ###
        #
        if moment2 == 0:
            return 0.0

        #
        return float(moment3 / (moment2**1.5))

    #
    def get_spectral_kurtosis(self) -> float:
        """Peakedness of the spectrum."""

        #
        mags, freqs = self._get_mag_spectrum()
        centroid = self.get_spectral_centroid()

        #
        sum_mag = np.sum(mags)

        #
        ### Avoid division by zero ###
        #
        if sum_mag == 0:
            return 0.0

        #
        moment4 = np.sum(((freqs - centroid) ** 4) * mags) / sum_mag
        moment2 = np.sum(((freqs - centroid) ** 2) * mags) / sum_mag

        #
        ### Avoid division by zero ###
        #
        if moment2 == 0:
            return 0.0

        #
        return float(moment4 / (moment2**2)) - 3.0

    #
    def get_spectral_slope(self) -> float:
        """Linear regression slope of the magnitude spectrum (log-log usually, but linear here for simplicity)."""

        #
        mags, freqs = self._get_mag_spectrum()

        #
        ### Use only up to 8kHz for slope to avoid high freq noise dominance ###
        #
        mask = freqs < 8000
        #
        if np.sum(mask) < 2:
            return 0.0

        #
        f_sub = freqs[mask]
        m_sub = mags[mask]

        #
        ### Normalize ###
        #
        f_norm = f_sub / np.max(f_sub)
        m_norm = m_sub / (np.max(m_sub) + 1e-10)

        #
        slope, _ = np.polyfit(f_norm, m_norm, 1)
        #
        return float(slope)

    #
    def get_spectral_flux(self) -> float:
        """
        Rate of change of the spectrum.
        Calculated by comparing frames.
        """

        #
        frame_size = 1024
        hop_size = 512
        #
        if len(self.audio_data) < frame_size:
            return 0.0

        #
        num_frames = (len(self.audio_data) - frame_size) // hop_size
        flux = 0.0
        prev_spectrum = np.zeros(frame_size // 2 + 1)

        #
        for i in range(num_frames):
            #
            start = i * hop_size
            frame = self.audio_data[start : start + frame_size]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))

            #
            ### Normalize ###
            #
            spectrum = spectrum / (np.sum(spectrum) + 1e-10)

            #
            diff = np.sum((spectrum - prev_spectrum) ** 2)
            flux += diff
            prev_spectrum = spectrum

        #
        return float(flux / num_frames)

    # --- 3. Harmonic Analysis ---

    #
    def get_fundamental_frequency(self) -> float:
        """
        Estimates F0 using autocorrelation.
        Returns 0.0 if no clear pitch found.
        """

        #
        ### Downsample for speed and robustness (e.g. to 11kHz) ###
        #
        target_sr = 11025
        #
        if self.sample_rate > target_sr:
            #
            step = self.sample_rate // target_sr
            sig = self.audio_data[::step]
            sr = self.sample_rate / step
        #
        else:
            #
            sig = self.audio_data
            sr = self.sample_rate

        #
        ### Autocorrelation ###
        #
        corr = np.correlate(sig, sig, mode="full")
        corr = corr[len(corr) // 2 :]

        #
        ### Find first peak after zero lag ###
        ### Ignore first few samples (very high freq noise) ###
        #
        min_lag = int(sr / 2000)  # Max 2kHz
        max_lag = int(sr / 40)  # Min 40Hz

        #
        if len(corr) < max_lag:
            return 0.0

        #
        window = corr[min_lag:max_lag]
        #
        if len(window) == 0:
            return 0.0

        #
        peak_idx = np.argmax(window) + min_lag

        #
        ### Refine with parabolic interpolation could be done here, but simple is okay ###
        #
        if corr[peak_idx] < 0.1 * corr[0]:  # Threshold for unvoiced
            return 0.0

        #
        return float(sr / peak_idx)

    #
    def get_harmonic_to_noise_ratio(self) -> float:
        """
        Estimates HNR.
        Ratio of autocorrelation peak energy to total energy (minus peak).
        """

        #
        f0 = self.get_fundamental_frequency()

        #
        ### Avoid division by zero ###
        #
        if f0 == 0:
            return 0.0

        #
        lag = int(self.sample_rate / f0)
        #
        if lag >= len(self.audio_data):
            return 0.0

        #
        ### Autocorrelation at lag ###
        ### We need normalized autocorrelation ###
        #
        corr = np.correlate(self.audio_data, self.audio_data, mode="full")
        corr = corr[len(corr) // 2 :]

        #
        r0 = corr[0]  # Total power
        #
        if r0 == 0:
            return 0.0

        #
        ### Find peak near lag ###
        #
        search_window = 10
        start = max(0, lag - search_window)
        end = min(len(corr), lag + search_window)
        #
        if start >= end:
            return 0.0

        #
        r_tau = np.max(corr[start:end])

        #
        ### HNR = R(tau) / (R(0) - R(tau)) ###
        #
        if r0 - r_tau <= 1e-10:
            return 100.0  # Infinite HNR

        #
        hnr = r_tau / (r0 - r_tau)
        #
        return float(10 * np.log10(hnr))

    #
    def get_odd_even_harmonic_ratio(self) -> float:
        """
        Ratio of energy in odd harmonics vs even harmonics.
        > 1.0 means odd dominant (Square/Clarinet like).
        < 1.0 means even dominant (Sawtooth/Violin like).
        """

        #
        f0 = self.get_fundamental_frequency()

        #
        ### Avoid division by zero ###
        #
        if f0 == 0:
            return 0.0

        #
        mags, freqs = self._get_mag_spectrum()

        #
        odd_energy = 0.0
        even_energy = 0.0

        #
        ### Sum up to 20 harmonics ###
        #
        for n in range(1, 21):
            #
            target_f = f0 * n
            #
            ### Find bin ###
            #
            idx = np.argmin(np.abs(freqs - target_f))
            #
            ### Sum energy in small window around bin ###
            #
            window_energy = np.sum(mags[max(0, idx - 2) : idx + 3] ** 2)

            #
            if n % 2 == 1:
                odd_energy += window_energy
            else:
                even_energy += window_energy

        #
        ### Avoid division by zero ###
        #
        if even_energy == 0:
            return 10.0

        #
        return float(odd_energy / even_energy)

    #
    def get_tristimulus(self) -> tuple[float, float, float]:
        """
        Tristimulus values (Pollard & Jansson).
        T1: Fundamental (n=1)
        T2: Middle harmonics (n=2,3,4)
        T3: High harmonics (n=5+)
        """

        #
        f0 = self.get_fundamental_frequency()

        #
        ### Avoid division by zero ###
        #
        if f0 == 0:
            return 0.0, 0.0, 0.0

        #
        mags, freqs = self._get_mag_spectrum()
        #
        total_harmonic_energy = 0.0

        #
        h_energies = []

        #
        for n in range(1, 21):
            #
            target_f = f0 * n
            #
            idx = np.argmin(np.abs(freqs - target_f))
            #
            e = np.sum(mags[max(0, idx - 2) : idx + 3] ** 2)
            #
            h_energies.append(e)
            #
            total_harmonic_energy += e

        #
        ### Avoid division by zero ###
        #
        if total_harmonic_energy == 0:
            return 0.0, 0.0, 0.0

        #
        t1 = h_energies[0] / total_harmonic_energy
        #
        t2 = sum(h_energies[1:4]) / total_harmonic_energy
        #
        t3 = sum(h_energies[4:]) / total_harmonic_energy

        #
        return float(t1), float(t2), float(t3)

    #
    def get_pitch_clarity(self) -> float:
        """
        Returns the normalized autocorrelation peak height.
        Close to 1.0 means very clear pitch. Low means noisy/inharmonic.
        """

        #
        f0 = self.get_fundamental_frequency()

        #
        ### Avoid division by zero ###
        #
        if f0 == 0:
            return 0.0

        #
        lag = int(self.sample_rate / f0)
        corr = np.correlate(self.audio_data, self.audio_data, mode="full")
        corr = corr[len(corr) // 2 :]

        #
        search_window = 10
        start = max(0, lag - search_window)
        end = min(len(corr), lag + search_window)

        #
        if start >= end:
            return 0.0

        #
        r_tau = np.max(corr[start:end])

        #
        return float(r_tau / corr[0])

#
### Import Modules. ###
#
import argparse
import sys
import numpy as np
from scipy.io import wavfile  # type: ignore

import nasong.core.vis as lv
import nasong.core.utils as li
import nasong.core.config as lc
import nasong.core.song as ls


def load_audio(input_path: str, sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    """
    Loads audio from a WAV file or generates it from a Python song description.
    Returns (audio_data, sample_rate).
    """
    if input_path.endswith(".wav"):
        try:
            sr, data = wavfile.read(input_path)
            # Convert to float32 normalized to [-1, 1] if necessary
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32767.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0

            # If stereo, take the first channel for simplicity
            if len(data.shape) > 1:
                data = data[:, 0]

            return data, sr
        except Exception as e:
            print(f"❌ Error loading WAV file: {e}")
            sys.exit(1)

    elif input_path.endswith(".py"):
        try:
            print(f"ℹ️ Generating audio from '{input_path}'...")
            sound_file_obj = li.import_module_from_filepath(filepath=input_path)
            duration = getattr(sound_file_obj, "duration")
            function_of_time = getattr(sound_file_obj, "song")

            song = ls.Song(
                config=lc.Config(
                    sample_rate=sample_rate,
                    total_duration=duration,
                    output_filename="temp_vis_output.wav",  # Not actually used for file I/O here
                ),
                value_of_time=function_of_time,
            )

            # Render directly to memory
            # We need to access the internal render logic or just use the public API
            # Since Song.export_to_wav writes to disk, let's just render the buffer manually
            # using the same logic as Song.render() but keeping it in memory.

            # Actually, Song doesn't expose a direct "get_buffer" method easily without running the whole render loop.
            # Let's just use the render method which returns the buffer.
            audio_buffer = song.render()
            return audio_buffer, sample_rate

        except Exception as e:
            print(f"❌ Error generating audio from Python file: {e}")
            sys.exit(1)
    else:
        print("❌ Unsupported file format. Please provide a .wav or .py file.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sound Visualization and Analysis Tool"
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Input file (WAV or Python song description)",
    )
    parser.add_argument(
        "-o", type=str, help="Output filename for plots (e.g., waveform.png)"
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["waveform", "spectrogram", "spectrum", "all"],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Perform analysis and print metrics"
    )
    parser.add_argument(
        "-s", type=int, default=44100, help="Sample rate (for generation)"
    )

    args = parser.parse_args()

    # Load Audio
    audio_data, sr = load_audio(args.i, args.s)
    print(
        f"✅ Loaded audio: {len(audio_data)} samples at {sr} Hz ({len(audio_data) / sr:.2f}s)"
    )

    # Analysis
    if args.analyze:
        analyzer = lv.AudioAnalyzer(audio_data, sr)
        print("\n📊 COMPREHENSIVE AUDIO REPORT")
        print("==============================")

        # 1. General & Dynamics
        print("\n🔹 1. GENERAL & DYNAMICS")
        print(f"  - Duration:            {len(audio_data) / sr:.4f} s")
        print(f"  - Effective Duration:  {analyzer.get_effective_duration():.4f} s")
        print(f"  - RMS Amplitude:       {analyzer.get_rms():.4f}")
        print(f"  - Peak Amplitude:      {analyzer.get_peak_amplitude():.4f}")
        crest, dyn_db = analyzer.get_dynamic_range()
        print(f"  - Dynamic Range:       {dyn_db:.2f} dB (Crest Factor: {crest:.2f})")

        # 2. Time Domain Envelope
        print("\n🔹 2. TIME DOMAIN & ENVELOPE")
        adsr = analyzer.get_adsr_estimates()
        print(
            f"  - ADSR Estimates:      A={adsr['attack']:.3f}s, D={adsr['decay']:.3f}s, S={adsr['sustain_level']:.2f}, R={adsr['release']:.3f}s"
        )
        print(f"  - Temporal Centroid:   {analyzer.get_temporal_centroid():.4f} s")
        print(f"  - Temporal Skewness:   {analyzer.get_temporal_skewness():.4f}")
        print(f"  - Temporal Kurtosis:   {analyzer.get_temporal_kurtosis():.4f}")

        # 3. Spectral Shape (Timbre)
        print("\n🔹 3. SPECTRAL SHAPE (TIMBRE)")
        print(
            f"  - Spectral Centroid:   {analyzer.get_spectral_centroid():.2f} Hz (Brightness)"
        )
        print(f"  - Spectral Bandwidth:  {analyzer.get_spectral_bandwidth():.2f} Hz")
        print(
            f"  - Spectral Rolloff:    {analyzer.get_spectral_rolloff():.2f} Hz (85% Energy)"
        )
        print(
            f"  - Spectral Flatness:   {analyzer.get_spectral_flatness():.4f} (0=Tone, 1=Noise)"
        )
        print(f"  - Spectral Slope:      {analyzer.get_spectral_slope():.2e}")
        print(f"  - Spectral Flux:       {analyzer.get_spectral_flux():.2e}")
        print(f"  - Spectral Skewness:   {analyzer.get_spectral_skewness():.4f}")
        print(f"  - Spectral Kurtosis:   {analyzer.get_spectral_kurtosis():.4f}")
        print(f"  - Zero Crossing Rate:  {analyzer.get_zero_crossing_rate():.4f}")

        # 4. Harmonic & Pitch
        print("\n🔹 4. HARMONIC & PITCH")
        f0 = analyzer.get_fundamental_frequency()
        print(f"  - Fundamental Freq:    {f0:.2f} Hz")
        print(
            f"  - Pitch Clarity:       {analyzer.get_pitch_clarity():.4f} (AutoCorr Peak)"
        )
        print(
            f"  - HNR:                 {analyzer.get_harmonic_to_noise_ratio():.2f} dB"
        )
        print(
            f"  - Odd/Even Ratio:      {analyzer.get_odd_even_harmonic_ratio():.4f} (>1 Odd dom, <1 Even dom)"
        )
        t1, t2, t3 = analyzer.get_tristimulus()
        print(
            f"  - Tristimulus:         T1={t1:.2f} (Fund), T2={t2:.2f} (2-4), T3={t3:.2f} (High)"
        )

        # 5. Issues
        print("\n⚠️ ISSUES DETECTION")
        clipping = analyzer.detect_clipping()
        dc_offset = analyzer.detect_dc_offset()
        silence = analyzer.detect_silence()

        print(
            f"  - Clipping (>0.99):    {clipping:.2f}% {'🔴' if clipping > 0 else '✅'}"
        )
        print(
            f"  - DC Offset:           {dc_offset:.4f} {'🔴' if abs(dc_offset) > 0.01 else '✅'}"
        )
        print(f"  - Silence (<-60dB):    {silence:.2f}%")
        print("==============================\n")

    # Plotting
    if args.plot:
        if args.plot == "waveform" or args.plot == "all":
            out_name = args.o if args.o else "waveform.png"
            if args.plot == "all":
                out_name = "waveform.png"
            lv.plot_waveform(audio_data, sr, filename=out_name, show=False)

        if args.plot == "spectrogram" or args.plot == "all":
            out_name = args.o if args.o else "spectrogram.png"
            if args.plot == "all":
                out_name = "spectrogram.png"
            lv.plot_spectrogram(audio_data, sr, filename=out_name, show=False)

        if args.plot == "spectrum" or args.plot == "all":
            out_name = args.o if args.o else "spectrum.png"
            if args.plot == "all":
                out_name = "spectrum.png"
            lv.plot_spectrum(audio_data, sr, filename=out_name, show=False)


if __name__ == "__main__":
    main()

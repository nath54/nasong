from typing import List, Dict, Any
import numpy as np
from .base import NoteDetector

try:
    import audioflux as af
except ImportError:
    af = None

class AudioFluxDetector(NoteDetector):
    """
    Note detection using AudioFlux (PEF or YIN algorithms).
    High-performance C/C++ core with Python wrapper.
    Typically monophonic.
    """

    def detect(self, audio_segment: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        if af is None:
            raise ImportError("AudioFlux is not installed. Please install 'audioflux'.")

        # AudioFlux expects contiguous C-order array
        audio_segment = np.ascontiguousarray(audio_segment, dtype=np.float32)

        algo_type = self.config.get('audioflux_type', 'PEF')
        min_freq = self.config.get('audioflux_min_freq', 50.0)
        max_freq = self.config.get('audioflux_max_freq', 2000.0)
        slide_length = self.config.get('audioflux_slide_length', 1024)
        radix2_exp = self.config.get('audioflux_radix2_exp', 12) # 2^12 = 4096

        # Pitch processing
        # Note: AudioFlux Pitch API varies by algorithm

        pitches = []
        times = []
        confidences = [] # PEF might not return confidence directly in same way as YIN

        if algo_type == 'YIN':
            # YIN implementation in AudioFlux
            pitch_obj = af.PitchYIN(
                samplate=sample_rate, 
                low_fre=min_freq, 
                high_fre=max_freq,
                radix2_exp=radix2_exp, # 2^12 = 4096
                slide_length=slide_length
            )
            fre_arr, val1_arr, val2_arr = pitch_obj.pitch(audio_segment)
            
            pitches = fre_arr
            num_frames = len(fre_arr)
            times = np.arange(num_frames) * (slide_length / sample_rate)
            confidences = val2_arr # approximation

        else: # PEF
            pitch_obj = af.PitchPEF(
                samplate=sample_rate, 
                low_fre=min_freq, 
                high_fre=max_freq,
                radix2_exp=radix2_exp,
                slide_length=slide_length
            )
            fre_arr = pitch_obj.pitch(audio_segment)
            
            pitches = fre_arr
            num_frames = len(fre_arr)
            times = np.arange(num_frames) * (slide_length / sample_rate)
            # PEF doesn't return confidence directly in pitch(), so we use dummy 1.0s or maybe magnitude if separate call
            confidences = np.ones_like(fre_arr)
        # Segmentation Logic (Simple voicing check)
        notes = []
        current_start = None
        current_pitch_accum = []

        # Threshold for voicing. 0 means unvoiced usually in AF results
        for i, f in enumerate(pitches):
            if f > 0: # Voiced
                if current_start is None:
                    current_start = times[i]
                current_pitch_accum.append(f)
            else:
                if current_start is not None:
                    # End note
                    duration = times[i] - current_start
                    if duration > 0.05:
                        mean_pitch = np.mean(current_pitch_accum)
                        notes.append({
                            'start_time': float(current_start),
                            'duration': float(duration),
                            'frequencies': [float(mean_pitch)],
                            'confidence': 1.0
                        })
                    current_start = None
                    current_pitch_accum = []

        if current_start is not None:
            duration = times[-1] - current_start
            if duration > 0.05:
                mean_pitch = np.mean(current_pitch_accum)
                notes.append({
                    'start_time': float(current_start),
                    'duration': float(duration),
                    'frequencies': [float(mean_pitch)],
                    'confidence': 1.0
                })

        return notes

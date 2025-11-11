

#
class Config:

    #
    def __init__(
        #
        self,
        #
        ### The sample rate (samples per second) is a standard for CD quality audio. ###
        #
        sample_rate: int = 44100,
        #
        ### The duration of the sound in seconds. ###
        #
        total_duration: float = 40.0,
        #
        ### The output filename. ###
        #
        output_filename: str = "generated_sine_wave.wav",
        #
    ) -> None:

        #
        ### The sample rate (samples per second) is a standard for CD quality audio. ###
        #
        self.sample_rate: int = sample_rate
        #
        ### The duration of the sound in seconds. ###
        #
        self.total_duration: float = total_duration
        #
        ### The output filename. ###
        #
        self.output_filename: str = output_filename

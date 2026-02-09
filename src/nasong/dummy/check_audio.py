import sounddevice as sd


def list_audio_devices():
    devices = sd.query_devices()
    default_out = sd.default.device[1]

    with open("audio_info_v2.txt", "w", encoding="utf-8") as f:
        f.write("\n--- Available Audio Devices ---\n")
        for i, dev in enumerate(devices):
            marker = ">> DEFAULT OUT <<" if i == default_out else ""
            f.write(
                f"[{i}] {dev['name']} (Channels: Out={dev['max_output_channels']}, In={dev['max_input_channels']}) {marker}\n"
            )

        f.write("\n--- Current Selection ---\n")
        if default_out == -1:
            f.write("Selected Output: None (Check system settings!)\n")
        else:
            current = devices[default_out]
            f.write(f"Index: {default_out}\n")
            f.write(f"Name: {current['name']}\n")
            f.write(f"Sample Rate: {current['default_samplerate']}\n")

        f.write(
            "\nNote: System-level volume cannot be directly retrieved via sounddevice.\n"
        )
        f.write("Please check your OS Volume Mixer.\n")


if __name__ == "__main__":
    list_audio_devices()

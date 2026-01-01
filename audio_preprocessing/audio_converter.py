from pydub import AudioSegment

def convert_to_wav_mono(audio : AudioSegment, target_sample_rate: int = 16000) -> AudioSegment:
    """Convert a stereo AudioSegment to mono with a specified sample rate.

    Args:
        audio (AudioSegment): The input stereo audio segment.
        sample_rate (int, optional): The desired sample rate. Defaults to 16000.

    Returns:
        AudioSegment: The converted mono audio segment.
    """

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sample_rate)
    audio = audio.set_sample_width(2)  # 16 bits = 2 bytes

    return audio
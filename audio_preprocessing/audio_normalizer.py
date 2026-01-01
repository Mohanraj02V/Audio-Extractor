from pydub import AudioSegment


def normalize_audio(audio: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:

    """Normalize the audio to a target dBFS level.

    Args:
        audio (AudioSegment): The input audio segment.
        target_dBFS (float, optional): The desired dBFS level. Defaults to -20.0.

    Returns:
        AudioSegment: The normalized audio segment.
    """

    change_in_dBFS = target_dBFS - audio.dBFS

    return audio.apply_gain(change_in_dBFS)
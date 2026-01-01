from pydub import AudioSegment

from utils.file_utils import validate_file_path


def load_audio(file_path: str) -> AudioSegment:
    """Load an audio file and return an AudioSegment object.

    Args:
        file_path (str): The path to the audio file.
    """

    validate_file_path(file_path)
    audio = AudioSegment.from_file(file_path)

    return audio
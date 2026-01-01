import os


SUPPORTED_FORMATS = (".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a")

def validate_file_path(file_path : str) -> None:
    """Validate if the given file path exists.

    Args:
        file_path (str): The path to the file to validate.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path {file_path} does not exist.")
    
    if not file_path.lower().endswith(SUPPORTED_FORMATS):
        raise ValueError(f"The file format of {file_path} is not supported. Supported formats are: {SUPPORTED_FORMATS}")
    
def create_dir_if_not_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)

import numpy as np


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert PCM16 little-endian bytes to float32 array in [-1.0, 1.0].

    Args:
        data: Raw PCM16 LE audio bytes (must have even length).

    Returns:
        Numpy float32 array normalised to [-1.0, 1.0].
    """
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def validate_audio(data: bytes) -> bool:
    """Check that *data* is valid PCM16 (even number of bytes, non-empty).

    Args:
        data: Raw audio bytes to validate.

    Returns:
        True when the data has even length and is non-empty.
    """
    if not data:
        return False
    return len(data) % 2 == 0

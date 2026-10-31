"""The binary audio frame of an audio session: a 2-byte big-endian stream number, then 16-bit little-endian mono
PCM at the session's sample rate, a whole number of samples. The header keeps the samples aligned, so a reader
can view them as 16-bit integers without copying."""

from __future__ import annotations

HEADER_BYTES = 2


def stream_of(frame: bytes) -> int:
    """The stream a frame belongs to: the session whose stream it is, on this connection."""
    return int.from_bytes(frame[:HEADER_BYTES], "big")


def samples_of(frame: bytes) -> bytes:
    """The samples of a frame, as 16-bit little-endian mono PCM."""
    return frame[HEADER_BYTES:]


def audio_frame(stream: int, samples: bytes) -> bytes:
    """A frame carrying `samples` for `stream`."""
    if len(samples) % 2:
        raise ValueError("audio is a whole number of 16-bit samples")
    return stream.to_bytes(HEADER_BYTES, "big") + samples


def bytes_per_frame(sample_rate: int, milliseconds: int = 20) -> int:
    """How many bytes of samples carry `milliseconds` of audio at `sample_rate`: 640 for 20 ms at 16000 Hz."""
    return sample_rate * milliseconds // 1000 * 2

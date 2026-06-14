from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg"}


def validate_audio_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported audio extension '{path.suffix}'. Try one of: {supported}")

    if path.stat().st_size == 0:
        raise ValueError("The uploaded audio file is empty.")


def maybe_convert_to_wav(path: Path, output_dir: Path) -> tuple[Path, str | None]:
    """Return a diarization-friendly 16 kHz mono WAV when ffmpeg is available."""
    validate_audio_path(path)

    if shutil.which("ffmpeg") is None:
        if path.suffix.lower() == ".wav":
            return path, "ffmpeg was not found, so the app used the uploaded WAV directly."
        return path, "ffmpeg was not found, so the app used the uploaded file directly."

    wav_path = output_dir / "audiomind_upload.16khz.mono.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return wav_path, "Normalized upload to 16 kHz mono WAV for more reliable diarization."

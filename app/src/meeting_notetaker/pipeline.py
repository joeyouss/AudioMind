from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class PipelineSettings:
    hf_token: str
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL
    whisper_model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    min_speakers: int | None = None
    max_speakers: int | None = None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def diarize(audio_path: Path, settings: PipelineSettings) -> list[dict]:
    from pyannote.audio import Pipeline

    if not settings.hf_token:
        raise ValueError("HF_TOKEN is required for pyannote diarization.")

    pipeline = Pipeline.from_pretrained(settings.pyannote_model, token=settings.hf_token)
    kwargs = {}
    if settings.min_speakers:
        kwargs["min_speakers"] = settings.min_speakers
    if settings.max_speakers:
        kwargs["max_speakers"] = settings.max_speakers

    diarization = pipeline(str(audio_path), **kwargs)
    return [
        {
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        }
        for turn, speaker in diarization.speaker_diarization
    ]


def transcribe(audio_path: Path, settings: PipelineSettings) -> list[dict]:
    from faster_whisper import WhisperModel

    device = resolve_device(settings.device)
    compute_type = resolve_compute_type(device, settings.compute_type)
    whisper = WhisperModel(settings.whisper_model, device=device, compute_type=compute_type)
    whisper_segments, info = whisper.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    for segment in whisper_segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append(
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "text": text,
                        "language": info.language,
                    }
                )
    return words


def who_spoke(word_start: float, word_end: float, speaker_segments: Iterable[dict]) -> str:
    midpoint = (word_start + word_end) / 2
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for segment in speaker_segments:
        overlap = min(word_end, segment["end"]) - max(word_start, segment["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment["speaker"]

        if segment["start"] <= midpoint <= segment["end"]:
            return segment["speaker"]

    return best_speaker


def align_words_to_speakers(words: Iterable[dict], speaker_segments: list[dict]) -> list[dict]:
    turns = []
    current = None

    for word in words:
        speaker = who_spoke(word["start"], word["end"], speaker_segments)

        if current is None or current["speaker"] != speaker:
            if current is not None:
                turns.append(current)
            current = {
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "text": word["text"],
            }
        else:
            current["end"] = word["end"]
            current["text"] += " " + word["text"]

    if current is not None:
        turns.append(current)

    return turns


def build_transcript(audio_path: Path, settings: PipelineSettings) -> dict:
    speaker_segments = diarize(audio_path, settings)
    words = transcribe(audio_path, settings)
    turns = align_words_to_speakers(words, speaker_segments)
    return {
        "audio_file": os.path.basename(audio_path),
        "speaker_segments": speaker_segments,
        "words": words,
        "turns": turns,
    }

from __future__ import annotations

import mimetypes
import time
import uuid
from pathlib import Path

import requests
import certifi
import urllib3


API_BASE = "https://api.pyannote.ai/v1"
REQUEST_VERIFY = certifi.where()


class PyannoteAPIError(RuntimeError):
    pass


def _verify_value(verify_ssl: bool):
    if verify_ssl:
        return REQUEST_VERIFY
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return False


def _auth_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _raise_for_status(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise PyannoteAPIError(f"{action} failed ({response.status_code}): {detail}") from exc


def check_api_key(api_key: str, verify_ssl: bool = True) -> dict:
    if not api_key.strip():
        return {"ok": False, "message": "Add a pyannoteAI API key to use Precision-2."}

    try:
        response = requests.get(
            f"{API_BASE}/test",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
            verify=_verify_value(verify_ssl),
        )
        if response.status_code == 200:
            return {"ok": True, "message": "pyannoteAI key is valid."}
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        return {"ok": False, "message": f"pyannoteAI key check failed ({response.status_code}): {detail}"}
    except requests.exceptions.SSLError as exc:
        return {
            "ok": False,
            "message": "SSL verification failed. Turn off 'Verify pyannoteAI SSL' for this local demo or fix local certificates.",
            "error": str(exc),
        }
    except requests.RequestException as exc:
        return {"ok": False, "message": f"Could not reach pyannoteAI: {exc}"}


def upload_media(path: Path, api_key: str, verify_ssl: bool = True) -> str:
    object_key = f"audiomind/{uuid.uuid4().hex}-{path.name}"
    media_url = f"media://{object_key}"

    response = requests.post(
        f"{API_BASE}/media/input",
        headers=_auth_headers(api_key),
        json={"url": media_url},
        timeout=60,
        verify=_verify_value(verify_ssl),
    )
    _raise_for_status(response, "Creating pyannoteAI media upload")
    upload_url = response.json()["url"]

    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as file:
        upload_response = requests.put(
            upload_url,
            data=file,
            headers={"Content-Type": content_type},
            timeout=600,
            verify=_verify_value(verify_ssl),
        )
    _raise_for_status(upload_response, "Uploading audio to pyannoteAI media storage")
    return media_url


def wait_for_job(
    job_id: str,
    api_key: str,
    poll_seconds: int = 5,
    timeout_seconds: int = 900,
    verify_ssl: bool = True,
) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(
            f"{API_BASE}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
            verify=_verify_value(verify_ssl),
        )
        _raise_for_status(response, "Checking pyannoteAI job")
        job = response.json()
        status = job["status"]

        if status == "succeeded":
            return job
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"pyannoteAI job {status}: {job}")

        time.sleep(poll_seconds)

    raise TimeoutError(f"pyannoteAI job timed out after {timeout_seconds} seconds: {job_id}")


def diarize_with_transcription(
    audio_url: str,
    api_key: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    verify_ssl: bool = True,
) -> dict:
    payload = {
        "url": audio_url,
        "model": "precision-2",
        "transcription": True,
    }
    if min_speakers:
        payload["minSpeakers"] = min_speakers
    if max_speakers:
        payload["maxSpeakers"] = max_speakers

    response = requests.post(
        f"{API_BASE}/diarize",
        headers=_auth_headers(api_key),
        json=payload,
        timeout=60,
        verify=_verify_value(verify_ssl),
    )
    _raise_for_status(response, "Starting Precision-2 transcription")
    job_id = response.json()["jobId"]
    return wait_for_job(job_id, api_key, verify_ssl=verify_ssl)


def create_voiceprint(sample_url: str, api_key: str, verify_ssl: bool = True) -> str:
    response = requests.post(
        f"{API_BASE}/voiceprint",
        headers=_auth_headers(api_key),
        json={"url": sample_url, "model": "precision-2"},
        timeout=60,
        verify=_verify_value(verify_ssl),
    )
    _raise_for_status(response, "Creating voiceprint")
    job_id = response.json()["jobId"]
    job = wait_for_job(job_id, api_key, verify_ssl=verify_ssl)
    return job["output"]["voiceprint"]


def identify_voice(audio_url: str, api_key: str, label: str, voiceprint: str, verify_ssl: bool = True) -> dict:
    response = requests.post(
        f"{API_BASE}/identify",
        headers=_auth_headers(api_key),
        json={
            "url": audio_url,
            "model": "precision-2",
            "voiceprints": [{"label": label, "voiceprint": voiceprint}],
            "matching": {"exclusive": True, "threshold": 0},
        },
        timeout=60,
        verify=_verify_value(verify_ssl),
    )
    _raise_for_status(response, "Starting voice identification")
    job_id = response.json()["jobId"]
    return wait_for_job(job_id, api_key, verify_ssl=verify_ssl)


def precision_turns(output: dict) -> list[dict]:
    turns = []
    for turn in output.get("turnLevelTranscription", []):
        turns.append(
            {
                "start": float(turn["start"]),
                "end": float(turn["end"]),
                "speaker": turn["speaker"],
                "text": turn["text"],
            }
        )
    return turns


def precision_words(output: dict) -> list[dict]:
    words = []
    for word in output.get("wordLevelTranscription", []):
        words.append(
            {
                "start": float(word["start"]),
                "end": float(word["end"]),
                "speaker": word.get("speaker", ""),
                "word": word.get("text", ""),
            }
        )
    return words


def apply_identified_speakers(turns: list[dict], identification_output: dict) -> list[dict]:
    segments = identification_output.get("diarization", [])
    if not segments:
        return turns

    labeled_turns = []
    for turn in turns:
        best_label = None
        best_overlap = 0.0
        for segment in segments:
            overlap = max(0.0, min(turn["end"], segment["end"]) - max(turn["start"], segment["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = segment.get("speaker")

        if best_label and best_overlap > 0:
            turn = {**turn, "speaker": best_label}
        labeled_turns.append(turn)
    return labeled_turns

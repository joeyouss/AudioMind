![AudioMind](static/AudioMind.png)

AudioMind turns community calls, conference conversations, office hours, and customer chats into speaker-aware transcripts, pain points, content ideas, and follow-up actions.

It is built for developers, DevRel teams, and community builders who want to mine useful signal from conversations without manually taking notes during every call.

With AudioMind, a developer can upload an audio file and get:

- a transcript with speaker labels
- timestamps for when each moment was spoken
- pain points and blockers
- content ideas with supporting evidence
- calls to action and follow-up tasks
- optional known-speaker labels using voiceprints
- downloadable transcript and insight JSON

## Why This Matters

Most useful community insight is hidden inside long conversations. People ask questions, describe blockers, suggest feature ideas, mention confusing docs, and volunteer follow-ups. Without a system like AudioMind, someone has to listen carefully, take notes, remember who said what, and later turn it into useful work.

Diarization is the part that answers: **who spoke when?**

Transcription answers: **what was said?**

Together, they let AudioMind connect an idea to the right speaker and timestamp. That makes the output much more useful than a plain transcript.

To directly see the Notebook with all the code and explanations, click [here](https://colab.research.google.com/drive/1r5jaUenFUPvNgnpS4Km8k69mleEb7Vz5?usp=sharing).

## Project Layout

```text
.
|-- README.md
|-- Makefile
|-- pyannote_community_notetaker.ipynb
|-- scripts/
|   `-- setup_local.sh
`-- app/
    |-- streamlit_app.py
    |-- requirements.txt
    |-- .env.example
    |-- .streamlit/
    |   `-- config.toml
    `-- src/meeting_notetaker/
        |-- audio.py
        |-- formatting.py
        |-- insights.py
        |-- pipeline.py
        `-- precision.py
```

## Prerequisites

Required for the recommended demo:

- Python 3.10 or 3.11
- a pyannoteAI API key
- Ollama for local Gemma analysis
- `ffmpeg` for reliable audio conversion

Required only for local pyannote + Whisper mode:

- a Hugging Face read token
- accepted access to `pyannote/speaker-diarization-community-1`

Recommended for demos:

- use a 1-5 minute audio clip first
- use WAV when possible
- use Precision-2 for the smoothest CPU-only demo
- use `tiny` or `base` Whisper if running the local mode on CPU

## Reproducible Local Run

These steps are the intended fresh-machine path for running AudioMind locally.

### 1. Clone the repo

```bash
git clone https://github.com/joeyouss/AudioMind.git
```

### 2. Install system prerequisites

AudioMind needs Python, `ffmpeg`, and Ollama.

On macOS with Homebrew:

```bash
brew install python@3.11 ffmpeg ollama
```

### 3. Run the project setup

```bash
make setup
```

This command:

- creates `app/.venv`
- installs all Python packages from `app/requirements.txt`
- creates `app/.env` from `app/.env.example`
- checks whether `ffmpeg` is available
- starts the Ollama server if it is not already running
- downloads the local Gemma model

By default, the app uses:

```bash
OLLAMA_MODEL=gemma3:1b
```

### 4. Add your API key

Open the environment file:

```bash
open app/.env
```

For the recommended demo path, add your pyannoteAI key:

```bash
PYANNOTE_API_KEY=your_pyannoteai_key_here
PYANNOTE_VERIFY_SSL=false
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
```

You cannot skip `PYANNOTE_API_KEY` for Precision-2 hosted mode because pyannoteAI needs to authenticate the API request.

Only add `HF_TOKEN` if you want to try the local pyannote + Whisper mode:

```bash
HF_TOKEN=hf_your_read_token_here
```

### 5. Start AudioMind

```bash
make run
```

Open:

```text
http://localhost:8501
```

Upload an audio file, keep `Precision-2 hosted` selected, and click **Analyze conversation**.

### 6. If Gemma/Ollama Does Not Connect

Start Ollama manually:

```bash
ollama serve
```

In another terminal, pull Gemma and restart the app:

```bash
ollama pull gemma3:1b
make run
```

### 7. Clean Restart

If the app gets into a bad local state, stop it with `Control-C` and run:

```bash
make run
```

If dependencies are broken, rebuild the local environment:

```bash
rm -rf app/.venv
make setup
make run
```

## Streamlit App

The Streamlit app supports:

- MP3, WAV, M4A, MP4, AAC, FLAC, and OGG uploads
- Precision-2 hosted diarization and transcription
- optional voiceprint identification in Precision-2 mode
- local pyannote + Whisper mode
- Gemma/Ollama analysis
- rule-based fallback insights if Ollama is unavailable
- transcript and JSON downloads

## Voiceprints

Voiceprints let AudioMind replace generic labels like `SPEAKER_00` with a known person when you provide a clean reference sample.

In the sidebar:

1. Choose `Precision-2 hosted`.
2. Turn on `Identify a known voice`.
3. Enter the known speaker name.
4. Upload a short sample where only that person speaks.
5. Upload the call audio.
6. Run analysis.

AudioMind creates a voiceprint from the sample, identifies that voice in the call, and updates transcript labels where there is a match.

## Notebook

Open:

```text
pyannote_community_notetaker.ipynb
```

The notebook is designed for a developer presentation. It explains:

- why diarization matters
- how speaker turns are created
- how transcription and speaker labels work together
- how Gemma reasons over timestamped transcript moments
- how Precision-2 compresses the pipeline into a smaller hosted API call
- how voiceprints can identify known speakers

In Colab, install dependencies with separate `pip` lines:

```python
!pip install -q --upgrade pip
!pip install -q pyannote.audio==4.0.4
!pip install -q faster-whisper==1.0.3
!pip install -q transformers==4.44.2
```

Restart the runtime after installing model libraries.

## Manual Setup

Use this if you do not want `make setup`:

```bash
cd app
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `app/.env`, then start Ollama:

```bash
ollama serve
```

In another terminal, start the app:

```bash
cd app
source .venv/bin/activate
ollama pull gemma3:1b
streamlit run streamlit_app.py
```

## Audio Notes

- WAV is the safest local format.
- MP3 and M4A usually work, but `ffmpeg` helps avoid decoding problems.
- Very short clips can produce weak speaker separation.
- Long calls should be tested after the workflow works on a short sample.
- For voiceprints, use a clean single-speaker sample.
- If local pyannote fails on an MP3, convert it to a 16 kHz mono WAV first.

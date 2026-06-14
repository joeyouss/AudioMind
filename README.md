# AudioMind

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

## How It Works

AudioMind supports two modes:

### 1. Precision-2 Hosted Mode

This is the recommended demo path.

Precision-2 runs through the pyannoteAI API and returns diarization plus transcription in one job. AudioMind uses the returned `turnLevelTranscription`, so you do not need to run Whisper separately or write custom word-to-speaker alignment code.

Use this mode when you want the simplest, cleanest developer demo.

### 2. Local pyannote + Whisper Mode

This mode shows the internals:

1. `pyannote.audio` detects speaker turns.
2. `faster-whisper` transcribes the audio with word timestamps.
3. AudioMind aligns words to speaker turns.
4. Gemma analyzes the transcript.

Use this mode when you want to teach how the pipeline works under the hood.

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

## Quickstart: Run Locally

Clone the repo, enter the project folder.

`make setup` does the local setup for you:

- creates `app/.venv`
- installs Python dependencies from `app/requirements.txt`
- creates `app/.env` from `app/.env.example` if it does not exist
- checks `ffmpeg`
- starts Ollama if needed
- pulls the local Gemma model from `OLLAMA_MODEL`

Now edit your local environment file:

```bash
open app/.env
```

For the recommended Precision-2 demo, set at least:

```bash
PYANNOTE_API_KEY=your_pyannoteai_key_here
PYANNOTE_VERIFY_SSL=false
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
```

If you also want to try local pyannote + Whisper mode, add:

```bash
HF_TOKEN=hf_your_read_token_here
```

Start the app:

```bash
make run
```

Open:

```text
http://localhost:8501
```

Upload an audio file, keep `Precision-2 hosted` selected, and click **Analyze conversation**.

### If Ollama Is Not Running

`make setup` tries to start Ollama automatically. If Gemma insights do not connect, run this in a separate terminal:

```bash
ollama serve
```

Then, in another terminal:

```bash
ollama pull gemma3:1b
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

## Troubleshooting

`pyannoteAI API key is missing`
: Add `PYANNOTE_API_KEY` to `app/.env` or paste it into the sidebar.

`pyannoteAI key check failed`
: Confirm the key is valid and that your network can reach `https://api.pyannote.ai`.

`SSL certificate verify failed`
: For local demos, keep `PYANNOTE_VERIFY_SSL=false`. For production, fix your local certificate store and turn verification back on.

`HF_TOKEN is missing`
: Add `HF_TOKEN` before using local pyannote + Whisper mode.

`401`, `403`, or gated model errors`
: Accept the Hugging Face model terms with the same account that owns your token.

`Ollama is not reachable`
: Start Ollama with `ollama serve`.

`Gemma model is not installed`
: Run `ollama pull gemma3:1b`.

`pip: command not found` in Colab
: Put each `!pip` command on its own line. Do not combine commands into one notebook line.

`CPU is slow`
: Use Precision-2 hosted mode, or choose `tiny` / `base` for local Whisper.

## Security

Never commit API keys, `.env` files, raw private calls, or generated transcripts that contain sensitive information.

`PYANNOTE_VERIFY_SSL=false` is included for local demo environments that have certificate issues. For production or shared deployments, use valid certificates and set SSL verification to true.

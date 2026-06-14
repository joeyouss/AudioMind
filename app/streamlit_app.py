from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(APP_DIR / "src"))

from meeting_notetaker.audio import maybe_convert_to_wav
from meeting_notetaker.formatting import to_pretty_json, transcript_to_markdown
from meeting_notetaker.insights import analyze_conversation, check_ollama
from meeting_notetaker.pipeline import PipelineSettings, build_transcript, resolve_compute_type, resolve_device
from meeting_notetaker.precision import (
    apply_identified_speakers,
    check_api_key,
    create_voiceprint,
    diarize_with_transcription,
    identify_voice,
    precision_turns,
    precision_words,
    upload_media,
)


load_dotenv()


st.set_page_config(
    page_title="AudioMind",
    page_icon="audio",
    layout="wide",
)

if "analysis_output" not in st.session_state:
    st.session_state.analysis_output = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None


st.markdown(
    """
    <style>
    :root {
        --black: #1A0007;
        --panel: #2A0811;
        --panel-soft: #350B16;
        --red: #D4142A;
        --red-deep: #9F1022;
        --text: #F7F7F7;
        --muted: #B9B9B9;
        --line: rgba(255, 255, 255, 0.16);
    }

    .stApp {
        background: var(--black);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.75rem;
        max-width: 1180px;
    }

    [data-testid="stSidebar"] {
        background: var(--red);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }

    [data-testid="stSidebar"] * {
        color: #fff;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.96);
        color: #111;
        border-color: rgba(255, 255, 255, 0.34);
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] input *,
    [data-testid="stSidebar"] textarea *,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * {
        color: #111 !important;
    }

    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button * {
        color: #fff !important;
    }

    [data-testid="stSidebar"] input::placeholder {
        color: rgba(17, 17, 17, 0.58);
    }

    h1, h2, h3, .stMarkdown strong {
        color: var(--text);
    }

    h1 {
        border-bottom: 4px solid var(--red);
        display: inline-block;
        padding-bottom: 0.35rem;
    }

    [data-testid="stCaptionContainer"],
    .small-muted {
        color: var(--muted);
    }

    .stButton > button,
    .stDownloadButton > button {
        background: var(--red);
        border: 1px solid var(--red);
        color: #fff;
        border-radius: 8px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--red-deep);
        border-color: var(--red-deep);
        color: #fff;
    }

    .stButton > button:disabled {
        background: rgba(212, 20, 42, 0.44);
        border-color: rgba(212, 20, 42, 0.1);
        color: rgba(255, 255, 255, 0.72);
    }

    [data-testid="stFileUploader"] section {
        background: var(--panel);
        border: 1px dashed rgba(255, 255, 255, 0.38);
        border-radius: 8px;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--red);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 4px solid var(--red);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--red);
        font-size: 1.35rem;
    }

    [data-testid="stTabs"] button {
        color: var(--text);
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--red);
        border-bottom-color: var(--red);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel-soft);
        border-color: var(--line);
        border-radius: 8px;
    }

    .stAlert {
        border-radius: 8px;
    }

    audio {
        border-radius: 8px;
        width: 100%;
    }

    .small-muted {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_items(title: str, items: list, empty: str) -> None:
    st.subheader(title)
    if not items:
        st.caption(empty)
        return
    for item in items:
        if isinstance(item, dict):
            heading = item.get("title") or item.get("question") or item.get("action") or item.get("task") or item.get("text")
            meta_parts = [
                item.get("speaker"),
                item.get("owner"),
                item.get("timestamp") or item.get("source_timestamp"),
                item.get("format"),
            ]
            meta = " | ".join(str(part) for part in meta_parts if part)
            with st.container(border=True):
                st.markdown(f"**{heading or 'Insight'}**")
                if meta:
                    st.caption(meta)
                for key, value in item.items():
                    if key in {"title", "question", "action", "task", "text", "speaker", "owner", "timestamp", "source_timestamp", "format"}:
                        continue
                    st.write(value)
        else:
            st.write(f"- {item}")


st.title("AudioMind")
st.caption("Upload a call, understand who said what, and turn the conversation into content ideas and follow-ups.")

with st.sidebar:
    st.header("Setup")
    engine = st.radio(
        "Transcript engine",
        ["Precision-2 hosted", "Local pyannote + Whisper"],
        index=0,
        key="transcript_engine_v2",
        help="Precision-2 returns diarization and transcription from pyannoteAI. Local mode runs pyannote + Whisper on this machine.",
    )
    use_precision = engine == "Precision-2 hosted"
    pyannote_api_key = ""
    pyannote_verify_ssl = True
    voiceprint_enabled = False
    known_speaker_name = ""
    voiceprint_sample = None

    if use_precision:
        pyannote_api_key = st.text_input(
            "pyannoteAI API key",
            value=os.environ.get("PYANNOTE_API_KEY", ""),
            type="password",
            help="Required for the hosted Precision-2 transcript path.",
        )
        pyannote_verify_ssl = st.checkbox(
            "Verify pyannoteAI SSL",
            value=os.environ.get("PYANNOTE_VERIFY_SSL", "false").lower() in {"1", "true", "yes"},
            help="Turn this on in normal environments. Leave off if local certificate verification fails during a demo.",
        )
        if not pyannote_verify_ssl:
            st.warning("SSL verification is off for pyannoteAI requests in this local session.")
        if pyannote_api_key:
            key_status = check_api_key(pyannote_api_key, verify_ssl=pyannote_verify_ssl)
            if key_status["ok"]:
                st.success(key_status["message"])
            else:
                st.error(key_status["message"])
        voiceprint_enabled = st.toggle("Identify a known voice", value=False)
        if voiceprint_enabled:
            known_speaker_name = st.text_input("Known speaker name", value="Known Speaker")
            voiceprint_sample = st.file_uploader(
                "Known speaker sample",
                type=["mp3", "wav", "m4a", "aac", "flac", "ogg"],
                help="Use a short clean clip where only this person speaks.",
            )
    else:
        pyannote_api_key = ""
        st.info("Local mode uses Hugging Face + local Whisper. Switch to Precision-2 hosted for the pyannoteAI API demo.")

    st.divider()
    st.header("Local settings")
    hf_token = st.text_input(
        "Hugging Face token",
        value=os.environ.get("HF_TOKEN", ""),
        type="password",
        help="Required for pyannote/speaker-diarization-community-1.",
        disabled=use_precision,
    )
    pyannote_model = st.text_input(
        "Pyannote model",
        value=os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"),
        disabled=use_precision,
    )
    whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=1, disabled=use_precision)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, disabled=use_precision)
    compute_type = st.selectbox("Compute type", ["auto", "int8", "float16", "float32"], index=0, disabled=use_precision)
    min_speakers = st.number_input("Minimum speakers", min_value=0, max_value=20, value=0)
    max_speakers = st.number_input("Maximum speakers", min_value=0, max_value=20, value=0)

    st.divider()
    st.header("Insights")
    use_ollama = st.toggle("Use Ollama/Gemma", value=True)
    ollama_url = st.text_input("Ollama URL", value=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ollama_model = st.text_input("Ollama model", value=os.environ.get("OLLAMA_MODEL", "gemma3:1b"))
    ollama_status = check_ollama(ollama_url, ollama_model)
    if ollama_status["ok"]:
        st.success(ollama_status["message"])
    else:
        st.warning(ollama_status["message"])

    if use_precision:
        st.caption("Runtime: hosted Precision-2 / hosted transcription")
    else:
        resolved_device = resolve_device(device)
        resolved_compute = resolve_compute_type(resolved_device, compute_type)
        st.caption(f"Runtime: {resolved_device} / {resolved_compute}")


uploaded = st.file_uploader("Upload MP3, WAV, M4A, FLAC, AAC, OGG, or MP4 audio", type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg"])

current_upload_signature = (uploaded.name, uploaded.size) if uploaded else None
if current_upload_signature != st.session_state.upload_signature:
    st.session_state.upload_signature = current_upload_signature
    st.session_state.analysis_output = None

if uploaded:
    st.audio(uploaded)

left, right = st.columns([1, 1])
with left:
    run = st.button("Analyze conversation", type="primary", disabled=uploaded is None)
with right:
    if use_precision:
        tip = "Tip: start with a 1-5 minute clip. Precision-2 returns speaker labels and transcript turns from one hosted job."
    else:
        tip = "Tip: start with a 2-5 minute clip. CPU demos are much faster with the tiny/base Whisper models."
    st.markdown(f'<p class="small-muted">{tip}</p>', unsafe_allow_html=True)


if run and uploaded:
    st.session_state.analysis_output = None

    if use_precision and not pyannote_api_key:
        st.error("Add a pyannoteAI API key in the sidebar before running Precision-2.")
        st.stop()

    if use_precision:
        key_status = check_api_key(pyannote_api_key, verify_ssl=pyannote_verify_ssl)
        if not key_status["ok"]:
            st.error(key_status["message"])
            st.stop()

    if not use_precision and not hf_token:
        st.error("Add a Hugging Face read token in the sidebar before running diarization.")
        st.stop()

    if voiceprint_enabled and not voiceprint_sample:
        st.error("Upload a known speaker sample or turn off voice identification.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        upload_path = tmpdir / uploaded.name
        upload_path.write_bytes(uploaded.getbuffer())

        try:
            with st.status("Preparing audio", expanded=True) as status:
                if use_precision:
                    st.write("Uploading audio to pyannoteAI media storage.")
                    audio_url = upload_media(upload_path, pyannote_api_key, verify_ssl=pyannote_verify_ssl)

                    st.write("Running Precision-2 diarization with hosted transcription.")
                    precision_job = diarize_with_transcription(
                        audio_url,
                        pyannote_api_key,
                        min_speakers=min_speakers or None,
                        max_speakers=max_speakers or None,
                        verify_ssl=pyannote_verify_ssl,
                    )
                    precision_output = precision_job["output"]
                    turns = precision_turns(precision_output)
                    words = precision_words(precision_output)
                    identification_output = None

                    if voiceprint_enabled and voiceprint_sample:
                        sample_path = tmpdir / voiceprint_sample.name
                        sample_path.write_bytes(voiceprint_sample.getbuffer())
                        st.write("Creating a voiceprint from the known speaker sample.")
                        sample_url = upload_media(sample_path, pyannote_api_key, verify_ssl=pyannote_verify_ssl)
                        voiceprint = create_voiceprint(sample_url, pyannote_api_key, verify_ssl=pyannote_verify_ssl)

                        st.write(f"Identifying {known_speaker_name} in the uploaded conversation.")
                        identification_job = identify_voice(
                            audio_url,
                            pyannote_api_key,
                            known_speaker_name,
                            voiceprint,
                            verify_ssl=pyannote_verify_ssl,
                        )
                        identification_output = identification_job["output"]
                        turns = apply_identified_speakers(turns, identification_output)

                    result = {
                        "turns": turns,
                        "words": words,
                        "raw_precision_output": precision_output,
                        "voiceprint_identification": identification_output,
                    }
                else:
                    settings = PipelineSettings(
                        hf_token=hf_token,
                        pyannote_model=pyannote_model,
                        whisper_model=whisper_model,
                        device=device,
                        compute_type=compute_type,
                        min_speakers=min_speakers or None,
                        max_speakers=max_speakers or None,
                    )
                    audio_path, conversion_note = maybe_convert_to_wav(upload_path, tmpdir)
                    if conversion_note:
                        st.write(conversion_note)
                    st.write("Running speaker diarization and Whisper transcription.")
                    result = build_transcript(audio_path, settings)

                st.write("Mining the conversation for content ideas and follow-ups.")
                insights = analyze_conversation(
                    result["turns"],
                    ollama_url=ollama_url,
                    model=ollama_model,
                    use_ollama=use_ollama,
                )
                status.update(label="Analysis complete", state="complete")
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.session_state.analysis_output = {
        "result": result,
        "insights": insights,
        "transcript_md": transcript_to_markdown(result["turns"]),
        "transcript_json": to_pretty_json(result["turns"]),
        "insights_json": to_pretty_json(insights),
    }

analysis_output = st.session_state.analysis_output

if analysis_output:
    result = analysis_output["result"]
    insights = analysis_output["insights"]
    transcript_md = analysis_output["transcript_md"]
    transcript_json = analysis_output["transcript_json"]
    insights_json = analysis_output["insights_json"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Speakers", len({turn["speaker"] for turn in result["turns"]}))
    metric_cols[1].metric("Transcript turns", len(result["turns"]))
    metric_cols[2].metric("Words", len(result["words"]))
    metric_cols[3].metric("Insight source", insights.get("source", "unknown"))

    if insights.get("ollama_error"):
        st.warning(f"Ollama was not available, so the app used fallback rules. Error: {insights['ollama_error']}")

    st.subheader("Brief")
    st.write(insights.get("summary") or "No summary generated.")

    tabs = st.tabs(["Transcript", "Pain Points", "Content Ideas", "Calls To Action", "Follow-ups", "FAQs", "Quotes", "Downloads"])

    with tabs[0]:
        st.markdown(transcript_md or "_No transcript turns were generated._")

    with tabs[1]:
        render_items("Pain Points", insights.get("pain_points", []), "No pain points found.")

    with tabs[2]:
        render_items("Content Ideas", insights.get("content_ideas", []), "No content ideas found.")

    with tabs[3]:
        render_items("Calls To Action", insights.get("calls_to_action", []), "No calls to action found.")

    with tabs[4]:
        render_items("Follow-up Tasks", insights.get("follow_up_tasks", []), "No follow-up tasks found.")

    with tabs[5]:
        render_items("FAQs", insights.get("faqs", []), "No FAQs found.")

    with tabs[6]:
        render_items("Quotes", insights.get("quotes", []), "No quotes found.")

    with tabs[7]:
        st.download_button("Download transcript.md", transcript_md, file_name="transcript.md", mime="text/markdown")
        st.download_button("Download transcript.json", transcript_json, file_name="transcript.json", mime="application/json")
        st.download_button("Download insights.json", insights_json, file_name="insights.json", mime="application/json")
        if result.get("raw_precision_output"):
            st.download_button(
                "Download precision2_output.json",
                to_pretty_json(result["raw_precision_output"]),
                file_name="precision2_output.json",
                mime="application/json",
            )
        if result.get("voiceprint_identification"):
            st.download_button(
                "Download voiceprint_identification.json",
                to_pretty_json(result["voiceprint_identification"]),
                file_name="voiceprint_identification.json",
                mime="application/json",
            )

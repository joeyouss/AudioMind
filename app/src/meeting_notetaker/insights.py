from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

import requests

from .formatting import mmss


INSIGHT_KEYS = [
    "summary",
    "pain_points",
    "content_ideas",
    "calls_to_action",
    "follow_up_tasks",
    "faqs",
    "quotes",
]


def check_ollama(ollama_url: str = "http://localhost:11434", model: str = "gemma3:1b") -> dict:
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=2)
        response.raise_for_status()
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Ollama is not reachable at {ollama_url}. Start Ollama before using Gemma.",
            "error": str(exc),
            "models": [],
        }

    models = response.json().get("models", [])
    model_names = sorted(model_info.get("name", "") for model_info in models)
    if model not in model_names:
        return {
            "ok": False,
            "message": f"Ollama is running, but {model} is not installed. Run: ollama pull {model}",
            "models": model_names,
        }

    return {
        "ok": True,
        "message": f"Gemma is ready: {model}",
        "models": model_names,
    }


def transcript_text(turns: Iterable[dict], limit: int = 12000) -> str:
    text = "\n".join(
        f'[{mmss(turn["start"])}] {turn["speaker"]}: {turn["text"]}'
        for turn in turns
    )
    return text[:limit]


def build_prompt(turns: list[dict]) -> str:
    return f"""
You are helping a DevRel and community team turn a conversation into useful work.

The input is a speaker-labeled transcript from a community call, conference hallway chat,
office-hours recording, or customer conversation.

Return only strict JSON with exactly this shape:
{{
  "summary": "2-4 sentence plain-language brief",
  "pain_points": [{{"speaker": "...", "timestamp": "MM:SS", "text": "...", "why_it_matters": "..."}}],
  "content_ideas": [{{"title": "...", "format": "blog|short video|workshop|docs|social post", "reason": "...", "source_timestamp": "MM:SS"}}],
  "calls_to_action": [{{"owner": "...", "action": "...", "source_timestamp": "MM:SS"}}],
  "follow_up_tasks": [{{"task": "...", "owner": "...", "source_timestamp": "MM:SS"}}],
  "faqs": [{{"question": "...", "answer_angle": "...", "source_timestamp": "MM:SS"}}],
  "quotes": [{{"speaker": "...", "timestamp": "MM:SS", "text": "..."}}]
}}

Look especially for:
- developer confusion, blockers, install problems, unclear docs, workflow friction
- content ideas a community team could publish
- explicit or implied next actions
- moments that should become onboarding, docs, demos, or event follow-ups

Transcript:
{transcript_text(turns)}
""".strip()


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(match.group(0))


def analyze_with_ollama(
    turns: list[dict],
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma3:1b",
    timeout: int = 600,
) -> dict:
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": build_prompt(turns)}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    insights = extract_json_object(content)
    return normalize_insights(insights, source=f"ollama:{model}")


def normalize_insights(data: dict, source: str) -> dict:
    normalized = {key: data.get(key, [] if key != "summary" else "") for key in INSIGHT_KEYS}
    normalized["source"] = source
    return normalized


def analyze_with_rules(turns: list[dict]) -> dict:
    pain_terms = [
        "blocked",
        "confused",
        "hard",
        "issue",
        "problem",
        "stuck",
        "can't",
        "cannot",
        "slow",
        "unclear",
        "error",
        "failed",
        "todo",
        "to-do",
    ]
    cta_terms = ["should", "need to", "follow up", "send", "create", "write", "ship", "publish"]
    question_turns = [turn for turn in turns if "?" in turn["text"]]
    pain_turns = [
        turn for turn in turns
        if any(term in turn["text"].lower() for term in pain_terms)
    ][:6]
    action_turns = [
        turn for turn in turns
        if any(term in turn["text"].lower() for term in cta_terms)
    ][:6]
    quote_turns = sorted(turns, key=lambda turn: len(turn["text"]), reverse=True)[:4]

    words = re.findall(r"[A-Za-z][A-Za-z0-9+'-]{3,}", " ".join(turn["text"] for turn in turns).lower())
    common = [
        word for word, _ in Counter(words).most_common(8)
        if word not in {"that", "this", "with", "have", "from", "they", "were", "what", "when", "there", "about"}
    ]
    topic = ", ".join(common[:4]) or "the conversation"

    return normalize_insights(
        {
            "summary": f"Rule-based fallback summary: the conversation appears to center on {topic}. Run Ollama for richer synthesis.",
            "pain_points": [
                {
                    "speaker": turn["speaker"],
                    "timestamp": mmss(turn["start"]),
                    "text": turn["text"],
                    "why_it_matters": "This sounds like friction worth turning into docs, demos, or follow-up support.",
                }
                for turn in pain_turns
            ],
            "content_ideas": [
                {
                    "title": f"Answer the community question from {mmss(turn['start'])}",
                    "format": "docs",
                    "reason": turn["text"],
                    "source_timestamp": mmss(turn["start"]),
                }
                for turn in question_turns[:5]
            ],
            "calls_to_action": [
                {
                    "owner": turn["speaker"],
                    "action": turn["text"],
                    "source_timestamp": mmss(turn["start"]),
                }
                for turn in action_turns
            ],
            "follow_up_tasks": [
                {
                    "task": turn["text"],
                    "owner": turn["speaker"],
                    "source_timestamp": mmss(turn["start"]),
                }
                for turn in action_turns[:4]
            ],
            "faqs": [
                {
                    "question": turn["text"],
                    "answer_angle": "Turn this into a short answer with a practical example.",
                    "source_timestamp": mmss(turn["start"]),
                }
                for turn in question_turns[:5]
            ],
            "quotes": [
                {
                    "speaker": turn["speaker"],
                    "timestamp": mmss(turn["start"]),
                    "text": turn["text"],
                }
                for turn in quote_turns
            ],
        },
        source="rule-based fallback",
    )


def analyze_conversation(turns: list[dict], ollama_url: str, model: str, use_ollama: bool) -> dict:
    if use_ollama:
        try:
            return analyze_with_ollama(turns, ollama_url=ollama_url, model=model)
        except Exception as exc:
            fallback = analyze_with_rules(turns)
            fallback["ollama_error"] = str(exc)
            return fallback
    return analyze_with_rules(turns)

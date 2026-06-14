from __future__ import annotations

import json
from typing import Iterable


def mmss(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def transcript_to_markdown(turns: Iterable[dict]) -> str:
    lines = []
    for turn in turns:
        lines.append(f'[{mmss(turn["start"])}] {turn["speaker"]}: {turn["text"]}')
    return "\n\n".join(lines) + ("\n" if lines else "")


def to_pretty_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


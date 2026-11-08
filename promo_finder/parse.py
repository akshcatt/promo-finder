"""Parse a timestamped microdrama screenplay into a flat list of timed events.

Pure code, no AI: the format is regular, so a parser is cheaper, faster and
exactly reproducible. Anything we can't parse is reported, not guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TS = r"(\d+):(\d+(?:\.\d+)?)"
LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SPAN_RE = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
SCENE_T_RE = re.compile(rf"^Scene (start|end):\s*{TS}")
NON_TAG = {"O.S.", "V.O.", "CONT'D", "O.C."}
TITLES = ("LADY ", "LORD ", "SIR ", "THE ")


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    """Seconds -> 'mm:ss.xx', the same format the screenplays use."""
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    ep: int
    idx: int
    kind: str  # camera | action | sfx | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""
    tags: list[str] = field(default_factory=list)
    scene: int = 0

    @property
    def dur(self) -> float:
        return self.end - self.start

    def short(self, n: int = 140) -> str:
        who = f"{self.speaker} ({', '.join(self.tags)})" if self.kind == "dialogue" else self.kind.upper()
        t = self.text if len(self.text) <= n else self.text[: n - 1] + "…"
        return f"[{fmt(self.start)}–{fmt(self.end)}] {who}: {t}"


@dataclass
class Episode:
    num: int
    path: str
    logline: str
    continuity: str
    events: list[Event]
    scenes: list[dict]
    warnings: list[str]

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)

    @property
    def dialogue(self) -> list[Event]:
        return [e for e in self.events if e.kind == "dialogue"]


def norm_speaker(raw: str) -> str:
    s = raw.upper().strip()
    for t in TITLES:
        if s.startswith(t):
            s = s[len(t):]
    return s.strip()


def parse_label(label: str) -> tuple[str, str, list[str]]:
    groups = re.findall(r"\(([^)]*)\)", label)
    head = re.split(r"\s*\(", label, maxsplit=1)[0].strip()
    up = head.upper()
    if up == "CAMERA":
        return "camera", "", []
    if up == "ACTION":
        return "action", "", []
    if up in ("SOUND FX", "SFX", "SOUND"):
        return "sfx", "", []
    tags: list[str] = []
    for g in groups:
        if g.strip().upper() in NON_TAG:
            continue
        tags += [t.strip().lower() for t in g.split(",") if t.strip()]
    return "dialogue", norm_speaker(head), tags


def parse_file(path: str | Path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    num_m = re.search(r"episode-(\d+)", path.name, re.I)
    num = int(num_m.group(1)) if num_m else 0
    logline = continuity = ""
    events: list[Event] = []
    scenes: list[dict] = []
    warnings: list[str] = []
    scene = 0
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("EPISODE NUMBER:"):
            m = re.search(r"(\d+)", ln)
            if m:
                num = int(m.group(1))
        elif ln.startswith("EPISODE LOGLINE:"):
            logline = ln.split(":", 1)[1].strip()
        elif ln.startswith("CONTINUITY NOTES:"):
            continuity = ln.split(":", 1)[1].strip()
        elif m := SCENE_RE.match(ln):
            scene = int(m.group(1))
            scenes.append({"scene": scene, "heading": "", "start": None, "end": None})
        elif ln.startswith("Scene heading:") and scenes:
            scenes[-1]["heading"] = ln.split(":", 1)[1].strip()
        elif (m := SCENE_T_RE.match(ln)) and scenes:
            scenes[-1][m.group(1)] = to_sec(m.group(2), m.group(3))
        elif m := LABEL_RE.match(ln):
            kind, speaker, tags = parse_label(m.group(1))
            span = SPAN_RE.match(lines[i + 1].strip()) if i + 1 < len(lines) else None
            if not span:
                warnings.append(f"line {i+1}: label without timestamps: {ln[:60]}")
                i += 1
                continue
            text_lines = []
            j = i + 2
            while j < len(lines) and lines[j].strip() and not LABEL_RE.match(lines[j].strip()):
                text_lines.append(lines[j].strip())
                j += 1
            text = " ".join(text_lines).strip().strip('"').strip("“”")
            s, e = to_sec(span.group(1), span.group(2)), to_sec(span.group(3), span.group(4))
            if e < s:
                warnings.append(f"line {i+1}: end before start, swapped")
                s, e = e, s
            events.append(Event(num, len(events), kind, s, e, text, speaker, tags, scene))
            i = j
            continue
        i += 1
    events.sort(key=lambda e: (e.start, e.end))
    for k, e in enumerate(events):
        e.idx = k
    if not events:
        warnings.append("no events parsed")
    return Episode(num, str(path), logline, continuity, events, scenes, warnings)


def load_show(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).glob("*.txt"))
    eps = [parse_file(f) for f in files]
    eps.sort(key=lambda e: e.num)
    return eps

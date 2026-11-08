"""Rule-based candidate generation and scoring (no AI, runs offline, ~1s per show).

Every number here is a tunable assumption, collected in CONFIG so an editor's
feedback can be turned into weight changes without touching logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Episode, Event, fmt

CONFIG = {
    "min_len": 20.0,
    "max_len": 60.0,
    "sweet_spot": (25.0, 45.0),     # most microdrama promos land here
    "hook_window": 3.0,             # brief: hook must land in first 3 seconds
    "cliff_window": 5.0,            # what the viewer is left with
    "weights": {"hook": 0.30, "cliff": 0.30, "intensity": 0.15, "standalone": 0.10, "plot": 0.15},
    "episode_end_bonus": 0.35,     # writers end every episode on their own cliffhanger
    "keep_per_episode": 4,          # after de-duplication
    "nms_overlap": 0.3,
}

# Emotion tags written by the screenwriters are the strongest cheap signal we have.
HOT_TAGS = {
    "screaming": 1.0, "shouting": 0.9, "furious": 1.0, "venomous": 0.9, "threatening": 0.9,
    "cruel": 0.8, "gloating": 0.8, "smug": 0.6, "mocking": 0.7, "taunting": 0.7, "spiteful": 0.7,
    "chilling": 0.8, "sinister": 0.8, "menacing": 0.9, "enraged": 1.0, "hysterical": 0.9,
    "desperate": 0.7, "pleading": 0.6, "sobbing": 0.7, "heartbroken": 0.7, "devastated": 0.8,
    "gasping": 0.6, "shocked": 0.8, "horrified": 0.9, "incredulous": 0.6, "accusing": 0.7,
    "accusatory": 0.7, "defiant": 0.7, "commanding": 0.6, "cold": 0.5, "scornful": 0.6,
    "contemptuous": 0.7, "arrogant": 0.5, "dismissive": 0.5, "harsh": 0.6, "fierce": 0.7,
    "unyielding": 0.5, "outraged": 0.9, "triumphant": 0.7, "resonant": 0.5, "booming": 0.7,
}
SOFT_TAGS = {"tender", "gentle", "soft", "reassuring", "calm", "warm", "loving", "sincere",
             "relieved", "content", "affectionate", "playful", "grateful", "thankful", "soothing"}

# Visual/verbal beats that stop a scroll. Weighted by how reliably they read on mute.
HOOK_WORDS = {
    r"\bslap": 1.0, r"\bstrik(e|es|ing)\b|\bstruck\b": 0.8, r"\bpunch": 0.8, r"\bexplo(de|sion)": 1.0,
    r"\bblood": 0.7, r"\bchain(s|ed)?\b": 0.7, r"\bknee(l|ls|ling)|\bon one knee": 0.6,
    r"\bfire\b|\bblaze|\bflames?\b|\binferno": 0.8, r"\bburn": 0.6, r"\bscream": 0.6,
    r"\bkiss": 0.6, r"\bthrow|\bthrew|\bhurl": 0.6, r"\bshov|\bgrab|\bseiz": 0.6,
    r"\bsword|\bblade|\bdagger|\bknife": 0.7, r"\bpoison": 0.8, r"\bwitch": 0.8,
    r"\bexecut": 0.9, r"\bdead\b|\bdie\b|\bkill": 0.8, r"\bduke\b|\bduchess\b": 0.6,
    r"\bwedding|\baltar|\bmarry|\bmarrying": 0.5, r"\bbetray": 0.7, r"\bhumiliat": 0.6,
    r"\bcrash|\bshatter": 0.7, r"\bguards?\b": 0.3, r"\bdrag": 0.6, r"\bgasp": 0.5,
    r"\bhot iron|\bred-hot|\bbrand": 0.9, r"\bdisown": 0.8, r"\bmud\b": 0.4, r"\bring\b": 0.3,
}
CLIFF_WORDS = {
    r"\bwho are you\b": 1.0, r"\bwill (pay|burn|die|hang|regret)": 1.0, r"\bnever\b": 0.4,
    r"\bdead\b|\bdie\b|\bkill": 0.7, r"\bexecut": 0.8, r"\bmine\b": 0.4, r"\btruth\b": 0.5,
    r"\bduke\b|\bduchess\b": 0.6, r"\bburn": 0.6, r"\bwitch": 0.6, r"\bpoison": 0.6,
    r"\bsecret": 0.6, r"\bwhat have you done\b": 1.0, r"\bhow dare\b": 0.6, r"\bguards\b": 0.6,
    r"\barrest|\bseize": 0.6, r"\bkneel\b": 0.6, r"\bwait\b": 0.3, r"\bfrozen|\bfreez": 0.3,
}
CLIFF_SFX = re.compile(r"cliffhanger|ominous|swell|sting|stinger|thunder|gasp|boom|explosion|"
                       r"crash|shatter|scream|drone|heartbeat|silence|cut to black|smash", re.I)
DEAD_OPEN = re.compile(r"establishing|aerial|drone shot|exterior|wide shot of the|landscape|"
                       r"fade in|title card|skyline|time-?lapse", re.I)
CONTEXT_NEEDY = re.compile(r"\b(remember|again|earlier|yesterday|last night|as I said|"
                           r"like I told|you promised|that day)\b", re.I)
MID_CONVO_OPEN = re.compile(r"^(and|but|so|then|because|which|that)\b", re.I)
RESOLVED = re.compile(r"\b(thank you|you're safe|it's over|all is well|I love you too|"
                      r"goodnight|rest now|we're home)\b", re.I)

ARCHETYPES = {
    "betrayal/humiliation": r"altar|jilt|disown|humiliat|betray|plain|worthless|mud|abandon|tolerate",
    "protector/rescue": r"protect|rescue|carr(y|ies)|punch|guards? collaps|step(s)? in front|shield|swear",
    "identity reveal": r"\bduke\b|crest|true (heir|identity)|signet|alive|regalia|rightful",
    "villain cruelty": r"hot iron|torment|whip|crop|scrub|starv|chain|cell|dungeon",
    "accusation/trial": r"witch|trial|poison|arrest|accus|testimony|execut|sentence",
    "romance": r"\bkiss|embrace|caress|\bI love you",
    "revenge/comeback": r"pay for|regret|kneel|proclaim|downfall|justice|reckoning",
}


STOP = set("""a an the and or but of to in on at by for with from as is are was were be been her his him
she he they them their its it this that these those into onto over under after before while when where who whom
which what than then only just even also new old own very most more much one two three not no yet so up out down
off again ever every all any some such her's himself herself lady lord sir""".split())


def content_words(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def wscore(text: str, table: dict[str, float]) -> float:
    return sum(w for pat, w in table.items() if re.search(pat, text, re.I))


def tag_heat(e: Event) -> float:
    return max((HOT_TAGS.get(t, 0.0) for t in e.tags), default=0.0)


def is_soft(e: Event) -> bool:
    return any(t in SOFT_TAGS for t in e.tags)


@dataclass
class Window:
    ep: int
    start: float
    end: float
    events: list[Event]
    features: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    score: float = 0.0

    @property
    def dur(self) -> float:
        return self.end - self.start

    def key(self) -> str:
        return f"E{self.ep:02d}@{fmt(self.start)}-{fmt(self.end)}"


def cut_points(ep: Episode) -> tuple[list[float], list[float]]:
    """Legal in/out points: shot boundaries that don't slice through a spoken line."""
    cams = [e for e in ep.events if e.kind == "camera"] or ep.events
    lines = ep.dialogue

    def clean(t: float) -> bool:
        return not any(d.start + 0.05 < t < d.end - 0.05 for d in lines)

    ins = sorted({round(c.start, 2) for c in cams if clean(c.start)})
    outs = sorted({round(c.end, 2) for c in cams if clean(c.end)})
    return ins, outs


def score_window(w: Window, ep: Episode, cfg=CONFIG) -> None:
    ev = w.events
    s, e = w.start, w.end
    hw, cw = cfg["hook_window"], cfg["cliff_window"]
    dlg = [x for x in ev if x.kind == "dialogue"]

    # ---- HOOK: what happens in the first 3 seconds
    head = [x for x in ev if x.start < s + hw]
    head_txt = " ".join(x.text for x in head)
    first_line = next((x for x in dlg if x.start < s + hw), None)
    hook = 0.0
    hook += min(wscore(head_txt, HOOK_WORDS), 1.5) / 1.5 * 0.45
    if first_line:
        hook += 0.25 + 0.30 * tag_heat(first_line)
        if first_line.start - s > 1.5:
            hook -= 0.10  # silence before the first line loses the scroll
        if MID_CONVO_OPEN.match(first_line.text):
            hook -= 0.15
    else:
        hook += 0.05 if any(x.kind == "action" for x in head) else 0.0
    first_cam = next((x for x in head if x.kind == "camera"), None)
    if first_cam and DEAD_OPEN.search(first_cam.text):
        hook -= 0.25

    # ---- CLIFFHANGER: what the viewer is left with
    tail = [x for x in ev if x.end > e - cw]
    tail_txt = " ".join(x.text for x in tail)
    last_line = dlg[-1] if dlg else None
    cliff = 0.0
    cliff += min(wscore(tail_txt, CLIFF_WORDS), 1.5) / 1.5 * 0.35
    if any(x.kind == "sfx" and CLIFF_SFX.search(x.text) for x in tail):
        cliff += 0.20
    at_ep_end = e >= ep.duration - 0.5
    if at_ep_end:
        cliff += cfg["episode_end_bonus"]  # writers built the episode ending as a cliffhanger
    if last_line and last_line.end > e - cw:
        t = last_line.text.rstrip()
        if t.endswith("?"):
            cliff += 0.20
        elif t.endswith(("...", "…", "—", "-")):
            cliff += 0.15
        cliff += 0.15 * tag_heat(last_line)
        if is_soft(last_line) or RESOLVED.search(t):
            cliff -= 0.25
    nxt = next((x for x in ep.dialogue if x.start >= e - 0.05), None)
    if nxt and nxt.start - e < 2.0 and not at_ep_end:
        cliff += 0.05  # cutting right before the reply

    # ---- INTENSITY: is the middle worth watching
    spoken = sum(x.dur for x in dlg)
    coverage = min(spoken / max(w.dur, 1), 1.0)
    heat = sum(tag_heat(x) for x in dlg) / max(len(dlg), 1)
    soft_share = sum(is_soft(x) for x in dlg) / max(len(dlg), 1)
    body_hits = min(wscore(" ".join(x.text for x in ev), HOOK_WORDS), 4) / 4
    intensity = 0.35 * coverage + 0.35 * heat + 0.30 * body_hits - 0.25 * soft_share

    # ---- STANDALONE: will a stranger follow it
    speakers = {x.speaker for x in dlg}
    standalone = 0.6
    if len(speakers) == 0:
        standalone -= 0.4
    elif len(speakers) == 1:
        standalone -= 0.1
    elif len(speakers) >= 6:
        standalone -= 0.2
    needy = sum(bool(CONTEXT_NEEDY.search(x.text)) for x in dlg)
    standalone -= min(0.1 * needy, 0.3)
    if any(abs(s - (sc.get("start") or -9)) < 0.3 for sc in ep.scenes):
        standalone += 0.2  # entering at a scene start = natural setup
    if ep.num <= 3:
        standalone += 0.1  # premise episodes explain themselves
    if len(dlg) >= 4:
        standalone += 0.1

    # ---- PLOT: does the clip contain what the writers say the episode is about?
    # The logline is the writers' own one-line summary of the key beat; free signal.
    lw = content_words(ep.logline)
    ww = content_words(" ".join(x.text for x in ev))
    plot = len(lw & ww) / max(min(len(lw), 12), 1) * 1.6

    lo, hi = cfg["sweet_spot"]
    len_pen = 0.0 if lo <= w.dur <= hi else 0.05

    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    w.scores = {"hook": clamp(hook), "cliff": clamp(cliff),
                "intensity": clamp(intensity), "standalone": clamp(standalone),
                "plot": clamp(plot)}
    W = cfg["weights"]
    w.score = round(sum(W[k] * w.scores[k] for k in W) - len_pen, 4)
    w.features = {
        "first_line": first_line.text if first_line else "",
        "first_line_t": round(first_line.start - s, 2) if first_line else None,
        "first_line_tags": first_line.tags if first_line else [],
        "last_line": last_line.text if last_line else "",
        "last_line_tags": last_line.tags if last_line else [],
        "ends_at_episode_end": at_ep_end,
        "dialogue_coverage": round(coverage, 2),
        "speakers": sorted(speakers),
        "dead_open": bool(first_cam and DEAD_OPEN.search(first_cam.text)),
    }


def archetype(text: str) -> str:
    best, n = "drama", 0
    for name, pat in ARCHETYPES.items():
        k = len(re.findall(pat, text, re.I))
        if k > n:
            best, n = name, k
    return best


def overlap(a: Window, b: Window) -> float:
    if a.ep != b.ep:
        return 0.0
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / min(a.dur, b.dur)


def nms(ws: list[Window], thr: float, k: int) -> list[Window]:
    kept: list[Window] = []
    for w in sorted(ws, key=lambda x: -x.score):
        if all(overlap(w, x) < thr for x in kept):
            kept.append(w)
        if len(kept) >= k:
            break
    return kept


def windows_for_episode(ep: Episode, cfg=CONFIG, min_len=None, max_len=None) -> list[Window]:
    lo = cfg["min_len"] if min_len is None else min_len
    hi = cfg["max_len"] if max_len is None else max_len
    ins, outs = cut_points(ep)
    out: list[Window] = []
    for s in ins:
        for e in outs:
            if lo <= e - s <= hi:
                ev = [x for x in ep.events if x.start >= s - 0.05 and x.end <= e + 0.05]
                w = Window(ep.num, s, e, ev)
                score_window(w, ep, cfg)
                out.append(w)
    return out


def shortlist(ep: Episode, cfg=CONFIG) -> list[Window]:
    return nms(windows_for_episode(ep, cfg), cfg["nms_overlap"], cfg["keep_per_episode"])

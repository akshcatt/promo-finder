"""Multi-segment (stitched) promos, final ranking, safety flags, and posting order."""
from __future__ import annotations

import re

from .heuristics import CONFIG, Window, archetype, nms, windows_for_episode
from .parse import Episode, fmt

# Platform-policy screen. A hit doesn't drop the promo, it routes it to a human.
UNSAFE = re.compile(r"\b(sex|sexual|naked|nude|undress\w*|bare[- ](skin|chest\w*|back)|chemise|straddl\w*|moan\w*|"
                    r"lingerie|bedsheets?|sheets pulled|in (the|this|my|your) bed|intimacy|illicit|"
                    r"hot iron|branding|brand(s|ed)? (her|his)|blood (pours|gushes|spurts)|gore|self-harm|suicide)\b", re.I)


LOGLINES: dict[int, str] = {}  # filled by run.py so flags can use episode-level context
STITCH = {"part_min": 8.0, "part_max": 25.0, "max_gap_eps": 2, "keep": 12}


def safety_flags(w_events, logline: str = "") -> list[str]:
    hits = sorted({m.group(0).lower() for e in w_events for m in [UNSAFE.search(e.text)] if m})
    if logline and (m := UNSAFE.search(logline)):
        hits.append(f"episode logline: {m.group(0).lower()}")
    return [f"brand-safety review: '{h}'" for h in hits]


def stitched_candidates(eps: list[Episode], cfg=CONFIG) -> list[dict]:
    """Opener (strongest hook) from one episode + closer (strongest cliffhanger) from the
    same or a nearby episode. Assumes nearby episodes share a story arc, which the pack
    doesn't always honour, so every stitch is flagged for an editor."""
    openers, closers = [], []
    for ep in eps:
        parts = windows_for_episode(ep, cfg, STITCH["part_min"], STITCH["part_max"])
        for p in parts:
            p.score = p.scores["hook"] + 0.3 * p.scores["intensity"]
        openers += nms(parts, 0.3, 2)
        for p in parts:
            p.score = p.scores["cliff"] + 0.3 * p.scores["intensity"]
        closers += nms(parts, 0.3, 2)
    out = []
    for o in openers:
        for c in closers:
            gap = c.ep - o.ep
            if not (0 <= gap <= STITCH["max_gap_eps"]):
                continue
            if o.ep == c.ep and not (o.end <= c.start):
                continue  # same episode: keep chronological, no overlap
            total = o.dur + c.dur
            if not (cfg["min_len"] <= total <= cfg["max_len"]):
                continue
            W = cfg["weights"]  # same scale as single clips, minus a cost for joining
            sc = {"hook": o.scores["hook"], "cliff": c.scores["cliff"],
                  "intensity": (o.scores["intensity"] + c.scores["intensity"]) / 2,
                  "standalone": min(o.scores["standalone"], c.scores["standalone"]),
                  "plot": max(o.scores["plot"], c.scores["plot"])}
            score = sum(W[k] * sc[k] for k in W) - 0.05 - 0.03 * gap
            out.append({"segments": [o, c], "score": round(score, 4), "gap": gap})
    out.sort(key=lambda d: -d["score"])
    picked, used = [], set()
    for d in out:
        k1 = (d["segments"][0].ep, d["segments"][0].start)
        k2 = (d["segments"][1].ep, d["segments"][1].start)
        if k1 in used or k2 in used:
            continue
        used |= {k1, k2}
        picked.append(d)
        if len(picked) >= STITCH["keep"]:
            break
    return picked


def to_promo(segs: list[Window], score: float, kind: str, scores: dict, extra: dict | None = None) -> dict:
    ev = [e for s in segs for e in s.events]
    first, last = segs[0], segs[-1]
    llm = (extra or {}).get("llm") or first.features.get("llm")
    hook = llm["hook"] if llm else _hook_text(first)
    cliff = llm["cliffhanger"] if llm else _cliff_text(last)
    why = llm["why"] if llm else _why(first, last, scores)
    flags = safety_flags(ev, " ".join(LOGLINES.get(s.ep, "") for s in segs))
    if kind == "stitched":
        flags.append("stitched across moments: editor to confirm continuity (names/plot drift in this pack)")
    if first.features.get("dead_open"):
        flags.append("opens on an establishing shot")
    if not last.features.get("last_line"):
        flags.append("no dialogue at the end; check visually")
    if first.features.get("backup"):
        flags.append("rules-only backup: the model did not pick this clip")
    if llm and llm.get("issues"):
        flags.append(f"model: {llm['issues']}")
    return {
        "kind": kind,
        "episodes": sorted({s.ep for s in segs}),
        "segments": [{"episode": s.ep, "start": fmt(s.start), "end": fmt(s.end),
                      "start_s": round(s.start, 2), "end_s": round(s.end, 2), "dur_s": round(s.dur, 2)} for s in segs],
        "total_s": round(sum(s.dur for s in segs), 2),
        "score": round(score, 4),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "hook": hook,
        "cliffhanger": cliff,
        "why": why,
        "archetype": archetype(" ".join(e.text for e in ev)),
        "flags": flags,
        "judged_by": f"llm({llm.get('model', '?')})+rules" if llm else "rules",
        "transcript": [e.short(160) for e in ev if e.kind != "camera"],
    }


def _hook_text(w: Window) -> str:
    f = w.features
    if f.get("first_line"):
        tags = f", {'/'.join(f['first_line_tags'][:2])}" if f.get("first_line_tags") else ""
        return f"\"{f['first_line']}\" (at +{f['first_line_t']:.1f}s{tags})"
    act = next((e for e in w.events if e.kind in ("action", "sfx")), None)
    return f"[visual] {act.text[:120]}" if act else "[visual opening, no line]"


def _cliff_text(w: Window) -> str:
    f = w.features
    s = f"\"{f['last_line']}\"" if f.get("last_line") else "[visual ending]"
    if f.get("ends_at_episode_end"):
        s += " (the episode's own ending sting)"
    return s


def _why(first: Window, last: Window, sc: dict) -> str:
    parts = []
    ff, lf = first.features, last.features
    if sc.get("hook", 0) >= 0.6:
        tag = (ff.get("first_line_tags") or ["charged"])[0]
        t0 = ff.get("first_line_t") or 0
        parts.append(f"opens instantly on a {tag} line" if t0 < 0.5 else f"{tag} line lands at {t0:.1f}s")
    if sc.get("cliff", 0) >= 0.5:
        lt = lf.get("last_line", "")
        parts.append("cuts on an unanswered question" if lt.endswith("?") else
                     "cuts on the episode's cliffhanger" if lf.get("ends_at_episode_end") else "cuts before the payoff")
    if sc.get("intensity", 0) >= 0.45:
        parts.append(f"{int(100 * ff.get('dialogue_coverage', 0))}% wall-to-wall confrontation")
    if sc.get("standalone", 0) >= 0.7:
        parts.append(f"{len(ff.get('speakers', []))} clear characters, no backstory needed")
    return "; ".join(parts) or "best available moment in a quiet episode"


def rank(single: list[Window], stitched: list[dict]) -> list[dict]:
    promos = [to_promo([w], w.score, "single", w.scores, {"llm": w.features.get("llm")}) for w in single]
    for d in stitched:
        o, c = d["segments"]
        sc = {"hook": o.scores["hook"], "cliff": c.scores["cliff"],
              "intensity": (o.scores["intensity"] + c.scores["intensity"]) / 2,
              "standalone": min(o.scores["standalone"], c.scores["standalone"]),
                  "plot": max(o.scores["plot"], c.scores["plot"])}
        promos.append(to_promo([o, c], d["score"], "stitched", sc))
    promos.sort(key=lambda p: -p["score"])
    for i, p in enumerate(promos, 1):
        p["quality_rank"] = i
    return promos


def posting_order(promos: list[dict], n: int = 10, total_eps: int = 50) -> list[dict]:
    """What to post first. Quality first, then two business rules:
    - funnel: a new viewer lands on episode 1, so clips from early episodes deliver
      what the ad promised soonest (ReelShort/DramaBox ads lean on the first ~10 eps);
    - variety: max 1 per episode and max 3 per archetype in the first batch, so the
      A/B test learns which angle (betrayal vs. rescue vs. reveal) works.
    Flagged-unsafe promos never go in the auto batch."""
    def adj(p):
        first_ep = min(p["episodes"])
        funnel = 1.12 if first_ep <= 10 else 1.05 if first_ep <= 20 else 1.0
        return p["score"] * funnel
    out, eps_used, arch = [], set(), {}
    for p in sorted(promos, key=lambda p: -adj(p)):
        if any(f.startswith("brand-safety") for f in p["flags"]):
            continue
        if any(e in eps_used for e in p["episodes"]) or arch.get(p["archetype"], 0) >= 3:
            continue
        out.append({**p, "post_score": round(adj(p), 4)})
        eps_used |= set(p["episodes"])
        arch[p["archetype"]] = arch.get(p["archetype"], 0) + 1
        if len(out) >= n:
            break
    return out

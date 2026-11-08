"""Optional LLM re-judging of the rule-based shortlist.

Anti-hallucination design: the model never writes timestamps. It sees numbered
cut points (C00, C01, ...) that code derived from the screenplay, and may only
answer with those IDs. Code maps IDs back to times and rejects anything that
is unknown, illegal (in/out role), or outside 20-60s. The model's job is the
part rules are bad at: does it make sense to a stranger, is the ending truly
unresolved, and why would someone stop scrolling.
"""
from __future__ import annotations

from .heuristics import CONFIG, Window, cut_points, score_window
from .llm import LLM, LLMError
from .parse import Episode, fmt

SYSTEM = """You are a senior promo editor for a vertical microdrama app (US audience).
You pick 20-60s clips from episodes to post as TikTok/Reels/Shorts ads that pull in NEW viewers.
A great promo: (1) HOOK in the first 3 seconds that stops a scroll (slap, reveal, shocking line; never an establishing shot),
(2) ends on a CLIFFHANGER (cut at an unresolved moment), (3) makes sense to someone who has never seen the show.
You may ONLY use the cut-point IDs given (e.g. "C07"). Never invent timestamps, lines or events.
Score harshly: 10 = would post today, 5 = usable filler, 1 = do not post. Reply with JSON only."""

PROMPT = """SHOW EPISODE {ep}. Logline (context for you only, the viewer will NOT see it): {logline}

TRANSCRIPT (times are mm:ss.xx). Lines starting with "C##" are legal cut points with allowed role IN and/or OUT.
{transcript}

CANDIDATE PROMOS proposed by a rule-based pre-filter:
{candidates}

For EACH candidate, judge it and, if you can make it clearly better, move its start/end to other legal cut points
(start must be an IN cut, end an OUT cut, length 20-60s). You may also add at most ONE new candidate (id "NEW")
if the pre-filter missed a clearly better moment in this episode.

Return JSON: {{"promos":[{{"id":"K1","start_cut":"C03","end_cut":"C11","hook":"<what happens / is said in first 3s>",
"cliffhanger":"<the unresolved moment it cuts on>","why":"<one line, why a stranger would watch to the end>",
"hook_score":1-10,"cliff_score":1-10,"standalone_score":1-10,"overall":1-10,"issues":"<what an editor should check, or empty>"}}]}}"""


def build_cut_table(ep: Episode) -> tuple[dict[str, tuple[float, set[str]]], str]:
    ins, outs = cut_points(ep)
    times = sorted(set(ins) | set(outs))
    cuts = {}
    for i, t in enumerate(times):
        roles = set()
        if t in ins:
            roles.add("IN")
        if t in outs:
            roles.add("OUT")
        cuts[f"C{i:02d}"] = (t, roles)
    by_time = {t: cid for cid, (t, _) in cuts.items()}
    rows, used = [], set()
    for e in ep.events:
        for t, cid in by_time.items():
            if cid not in used and t <= e.start + 0.01:
                rows.append(f"{cid} @ {fmt(t)} [{'/'.join(sorted(cuts[cid][1]))}]")
                used.add(cid)
        if e.kind == "camera":
            rows.append(f"  {fmt(e.start)} (shot) {e.text[:90]}")
        elif e.kind == "dialogue":
            rows.append(f"  {fmt(e.start)} {e.speaker} ({', '.join(e.tags)}): \"{e.text}\"")
        else:
            rows.append(f"  {fmt(e.start)} ({e.kind}) {e.text[:140]}")
    for t, cid in by_time.items():
        if cid not in used:
            rows.append(f"{cid} @ {fmt(t)} [{'/'.join(sorted(cuts[cid][1]))}]")
    return cuts, "\n".join(rows)


def _cid(cuts, t):
    return min(cuts, key=lambda c: abs(cuts[c][0] - t))


def judge_episode(llm: LLM, ep: Episode, shortlist: list[Window], cfg=CONFIG) -> tuple[list[Window], list[str]]:
    """Returns re-scored windows plus a log of anything rejected."""
    cuts, transcript = build_cut_table(ep)
    cand_txt = "\n".join(
        f"K{i+1}: {_cid(cuts, w.start)} -> {_cid(cuts, w.end)} ({fmt(w.start)}-{fmt(w.end)}, {w.dur:.1f}s)"
        for i, w in enumerate(shortlist))
    log: list[str] = []
    try:
        data = llm.complete_json(SYSTEM, PROMPT.format(ep=ep.num, logline=ep.logline,
                                                       transcript=transcript, candidates=cand_txt))
    except LLMError as e:
        log.append(f"E{ep.num:02d}: LLM failed ({e}); kept rule-based scores")
        return shortlist, log
    items = data.get("promos", []) if isinstance(data, dict) else data
    out: list[Window] = []
    seen = set()
    for it in items if isinstance(items, list) else []:
        try:
            sc, ec = str(it["start_cut"]).strip(), str(it["end_cut"]).strip()
            if sc not in cuts or ec not in cuts:
                raise ValueError(f"unknown cut id {sc}/{ec}")
            s, sroles = cuts[sc]
            e, eroles = cuts[ec]
            if "IN" not in sroles or "OUT" not in eroles:
                raise ValueError(f"illegal role {sc}->{ec}")
            if not (cfg["min_len"] - 0.01 <= e - s <= cfg["max_len"] + 0.01):
                raise ValueError(f"length {e - s:.1f}s outside 20-60s")
            if (s, e) in seen:
                continue
            seen.add((s, e))
            ev = [x for x in ep.events if x.start >= s - 0.05 and x.end <= e + 0.05]
            w = Window(ep.num, s, e, ev)
            score_window(w, ep, cfg)
            num = lambda k: max(1.0, min(10.0, float(it.get(k, 5))))  # noqa: E731
            w.features["llm"] = {
                "id": it.get("id"), "hook": it.get("hook", ""), "cliffhanger": it.get("cliffhanger", ""),
                "why": it.get("why", ""), "issues": it.get("issues", ""),
                "hook_score": num("hook_score"), "cliff_score": num("cliff_score"),
                "standalone_score": num("standalone_score"), "overall": num("overall"),
                "model": llm.last_model,
            }
            w.features["rule_score"] = w.score
            w.score = round(0.35 * w.score + 0.65 * num("overall") / 10, 4)
            out.append(w)
        except (KeyError, ValueError, TypeError) as err:
            log.append(f"E{ep.num:02d}: rejected model item {it.get('id', '?') if isinstance(it, dict) else '?'}: {err}")
    if not out:
        log.append(f"E{ep.num:02d}: no valid model items; kept rule-based scores")
        return shortlist, log
    # Keep rule picks the model didn't return (or returned invalid) as low-ranked backups:
    # the model is better at ranking, the rules are better at coverage, so use both.
    from .heuristics import overlap
    for w in shortlist:
        if all(overlap(w, x) < cfg["nms_overlap"] for x in out):
            w.features["rule_score"] = w.score
            w.features["backup"] = True
            w.score = round(0.5 * w.score, 4)
            out.append(w)
    return out, log

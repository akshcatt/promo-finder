#!/usr/bin/env python3
"""Score a run against hand-labelled gold moments, and against naive baselines.

  python3 eval/evaluate.py outputs/<show>/rules/promos.json "<show folder>"

A tool clip "hits" a gold moment if it is in the same episode and contains the
gold peak (the one beat that makes the moment promo-worthy).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promo_finder.heuristics import cut_points  # noqa: E402
from promo_finder.parse import load_show  # noqa: E402


def sec(t: str) -> float:
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def load_gold(folder: Path):
    moments, top = [], []
    for f in sorted(folder.glob("gold_*.json")):
        d = json.loads(f.read_text())
        for m in d["moments"]:
            moments.append({**m, "s": sec(m["start"]), "e": sec(m["end"]), "p": sec(m["peak"])})
        top += [{"episode": t["episode"], "s": sec(t["start"])} for t in d.get("top5", [])]
    return moments, top


def hits(clips, g):  # clips: list of (ep, s, e)
    return any(ep == g["episode"] and s - 0.5 <= g["p"] <= e + 0.5 for ep, s, e in clips)


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def spearman(x, y):
    rx = {v: i for i, v in enumerate(sorted(range(len(x)), key=lambda i: x[i]))}
    ry = {v: i for i, v in enumerate(sorted(range(len(y)), key=lambda i: y[i]))}
    return statistics.correlation([rx[i] for i in range(len(x))], [ry[i] for i in range(len(y))])


def main():
    run = json.loads(Path(sys.argv[1]).read_text())
    eps = {e.num: e for e in load_show(sys.argv[2])}
    gold, gtop = load_gold(Path(__file__).parent)
    if not gold:
        sys.exit("no eval/gold_*.json found. Gold labels quote the confidential scripts, so they ship with the raw outputs, "
                 "not the repo; copy them into eval/ to run this.")
    gold = [g for g in gold if g["episode"] in eps]
    promos = run["all_promos"]
    singles = [p for p in promos if p["kind"] == "single"]
    by_ep: dict[int, list] = {}
    for p in singles:  # all_promos is sorted by score
        sg = p["segments"][0]
        by_ep.setdefault(sg["episode"], []).append((sg["episode"], sg["start_s"], sg["end_s"]))
    all_segs = [(s["episode"], s["start_s"], s["end_s"]) for p in promos for s in p["segments"]]
    best_gold = {}
    for g in gold:
        if g["episode"] not in best_gold or g["strength"] > best_gold[g["episode"]]["strength"]:
            best_gold[g["episode"]] = g
    k = max(len(v) for v in by_ep.values())

    def report(name, ep_clips):
        r1 = sum(hits(ep_clips.get(e, [])[:1], g) for e, g in best_gold.items()) / len(best_gold)
        rk = sum(hits(ep_clips.get(g["episode"], []), g) for g in gold) / len(gold)
        strong = [g for g in gold if g["strength"] >= 4]
        rs = sum(hits(ep_clips.get(g["episode"], []), g) for g in strong) / max(len(strong), 1)
        print(f"{name:34s} top-1 hits best gold/ep {r1:5.0%} | any of top-{k} hits gold {rk:5.0%} | strong(4-5) {rs:5.0%}")
        return r1, rk, rs

    print(f"gold: {len(gold)} moments over {len(best_gold)} episodes; run mode: {run['meta']['mode']}\n")
    res = {"tool": report("TOOL (all 50)", by_ep)}
    full_gold, full_best = gold, best_gold
    for lo, hi, tag in ((1, 25, "TOOL eps 1-25 (tuning set)"), (26, 50, "TOOL eps 26-50 (held out)")):
        gold = [g for g in full_gold if lo <= g["episode"] <= hi]
        best_gold = {e: g for e, g in full_best.items() if lo <= e <= hi}
        if gold:
            report(tag, by_ep)
    gold, best_gold = full_gold, full_best

    # Baseline A: last ~40s of each episode (the writers' own cliffhanger)
    tail = {}
    for n, ep in eps.items():
        ins, _ = cut_points(ep)
        s = min(ins, key=lambda t: abs((ep.duration - 40) - t))
        tail[n] = [(n, s, ep.duration)]
    res["tail"] = report("BASELINE last-40s of episode", tail)

    # Baseline B: random legal windows, same count per episode, averaged over 300 seeds
    rng = random.Random(7)
    acc = [0.0, 0.0, 0.0]
    for _ in range(300):
        rnd = {}
        for n, ep in eps.items():
            ins, outs = cut_points(ep)
            pairs = [(s, e) for s in ins for e in outs if 20 <= e - s <= 60]
            rnd[n] = [(n, s, e) for s, e in rng.sample(pairs, min(k, len(pairs)))]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            r = report("", rnd)
        acc = [a + b / 300 for a, b in zip(acc, r)]
    print(f"{'BASELINE random legal windows':34s} top-1 hits best gold/ep {acc[0]:5.0%} | any of top-{k} hits gold {acc[1]:5.0%} | strong(4-5) {acc[2]:5.0%}")

    # Does the tool know which episodes are weak? (best tool score vs gold strength, per episode)
    best_score = {}
    for p in singles:
        e = p["segments"][0]["episode"]
        best_score[e] = max(best_score.get(e, 0), p["score"])
    common = sorted(set(best_score) & set(best_gold))
    rho = spearman([best_score[e] for e in common], [best_gold[e]["strength"] for e in common])
    print(f"\nEpisode-strength agreement (Spearman, tool best score vs gold strength): {rho:+.2f}")

    # Boundaries on hits
    ds, de, ious = [], [], []
    for g in gold:
        for c in by_ep.get(g["episode"], []):
            if hits([c], g):
                ds.append(abs(c[1] - g["s"])); de.append(abs(c[2] - g["e"])); ious.append(iou((c[1], c[2]), (g["s"], g["e"])))
                break
    if ds:
        print(f"Boundary error on hits: start {statistics.median(ds):.1f}s, end {statistics.median(de):.1f}s (median); IoU {statistics.median(ious):.2f}")

    # Ranking: are the labellers' overall top picks near the top of the tool's list?
    def found_rank(t):
        for i, p in enumerate(promos, 1):
            for s in p["segments"]:
                g = next((m for m in gold if m["episode"] == t["episode"] and abs(m["s"] - t["s"]) < 1), None)
                if g and s["episode"] == g["episode"] and s["start_s"] - .5 <= g["p"] <= s["end_s"] + .5:
                    return i
        return None
    ranks = [(t["episode"], found_rank(t)) for t in gtop]
    print("Labellers' top-10 picks -> tool quality rank:", ", ".join(f"E{e}:{r or 'miss'}" for e, r in ranks))
    in20 = sum(1 for _, r in ranks if r and r <= 20)
    print(f"  {in20}/{len(ranks)} of them inside the tool's top 20 of {len(promos)}")

    # Misses worth reading
    print("\nStrong gold moments the tool missed (any of its clips):")
    for g in gold:
        if g["strength"] >= 4 and not hits(by_ep.get(g["episode"], []), g) and not hits(all_segs, g):
            print(f"  E{g['episode']:02d} {g['start']}-{g['end']} peak {g['peak']}: {g['why'][:100]}")


if __name__ == "__main__":
    main()

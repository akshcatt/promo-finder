# promo-finder

Reads a folder of timestamped microdrama screenplays and returns ranked 20–60s promo clips: episode, start/end of every segment, hook, cliffhanger, one-line "why", and flags for anything an editor must check.

No dependencies beyond Python 3.10+ standard library. Runs fully offline in rules mode; any LLM can be plugged in with a flag.

## Run

```bash
# 1. Rules only: free, offline, ~15s for 50 episodes
python3 run.py "/path/to/show folder"

# 2. Rules shortlist + LLM judge (free tiers work)
cp .env.example .env             # put GEMINI_API_KEY=... (or GROQ/OPENROUTER/ANTHROPIC/OPENAI) in .env; it's gitignored
python3 run.py "/path/to/show folder" --provider gemini            # default model gemini-3.6-flash; if it returns 503 "high demand", pass --model gemini-3.5-flash
python3 run.py "/path/to/show folder" --provider groq --model llama-3.3-70b-versatile
python3 run.py "/path/to/show folder" --provider ollama            # local, no key

# Free tiers cap requests per model per day (Gemini: 20). Give a fallback chain:
python3 run.py "/path/to/show folder" --provider gemini --model gemini-3.6-flash,gemini-3-flash-preview,gemini-3.1-flash-lite

# 3. Any OpenAI-compatible endpoint (e.g. one Kuku provides)
python3 run.py "/path/to/show folder" --provider custom --base-url https://host/v1 --api-key-env MY_KEY --model my-model

# Useful flags
--episodes 1-10     run a subset        --post 15        size of the post-first batch
--per-episode 4     clips kept/episode  --no-stitch      single clips only
--min-interval 4    seconds between LLM calls (free-tier rate limits)
--reasoning-effort  low|medium|high|none (auto = low for gemini)   --cache-dir  where LLM answers are cached
```

Outputs go to `outputs/<show>/<mode>/`:

| File | For |
|---|---|
| `report.html` | Editor view: post-first batch, then every promo with scores, flags and the clip's transcript |
| `report.md` | Same, plain text |
| `promos.csv` | Paste into a sheet / tracker |
| `promos.json` | Everything, including config and LLM usage, for other tools |

Evaluate against hand-labelled moments:

```bash
python3 eval/evaluate.py outputs/<show>/rules/promos.json "/path/to/show folder"
python3 tests/mock_llm_test.py "/path/to/show folder"   # LLM path end-to-end against a fake server
```

## How it works

```
screenplay .txt ──parse──▶ timed events ──▶ every legal 20–60s window ──rules score──▶ top 4/episode
   (code)                   (code)            (cuts only on shot boundaries,           (hook, cliff,
                                               never inside a spoken line)              intensity, standalone, plot)
                                                                                            │
                                         optional LLM judge (picks only from numbered cut IDs; code validates)
                                                                                            │
                         stitched promos (best hook + best cliffhanger from nearby episodes)│
                                                                                            ▼
                          rank ──▶ brand-safety + continuity flags ──▶ post-first batch (funnel + variety rules)
```

| Step | Who does it | Why |
|---|---|---|
| Parse, legal cut points, 20–60s, no cut mid-line | Code | Deterministic; must never be wrong |
| Hook / cliffhanger / intensity / plot scoring | Rules (`heuristics.py`, all weights in `CONFIG`) | Free, instant, explainable, uses the writers' own emotion tags, SFX notes and logline |
| "Does a stranger get it? Is it really unresolved?" | LLM (optional) | Needs reading comprehension |
| Timestamps | Never the LLM | Model answers with cut IDs like `C07`; unknown/illegal IDs are rejected and logged |
| Coverage when the LLM skips a clip | Rules | Unpicked rule clips stay as low-ranked backups (rules = coverage, LLM = ranking) |
| Final pick, safety, posting | Human editor | Flags route risky clips to a person; nothing auto-posts |

## Files

```
run.py                     CLI
promo_finder/parse.py      screenplay -> events
promo_finder/heuristics.py candidate windows + rule scores (CONFIG holds every weight)
promo_finder/llm.py        provider-agnostic client (stdlib HTTP, disk cache, retries)
promo_finder/judge.py      LLM re-judging with cut-ID validation
promo_finder/assemble.py   stitched promos, ranking, safety flags, posting order
promo_finder/report.py     html / md / csv / json
eval/evaluate.py           recall vs gold + baselines
eval/gold_*.json           hand-labelled moments (70, all 50 episodes); quote the scripts, so shared with raw outputs, not in git
tests/mock_llm_test.py     LLM path test with a fake server (cut-ID validation + quota fallback)
package.sh                 builds ../submission/ (doc PDF, code without secrets, raw outputs zip)
```

The input screenplays are confidential and are not part of this repo.

## Results on the input pack (50 episodes, 70 gold moments)

| | Top pick hits best gold moment (held-out 26–50) | Any of 4 picks hits gold | Labellers' top 10 in tool's top 20 |
|---|---|---|---|
| Rules + Gemini | 80% | 93% | 4/10 |
| Rules only | 64% | 97% | 1/10 |
| Random clips | 33% (all eps) | 73% | – |

Full numbers and failures are in the solution doc (submitted separately; it quotes the confidential scripts).

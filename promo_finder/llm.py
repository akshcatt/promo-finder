"""Provider-agnostic LLM client. Standard library only (no SDKs to install).

Swapping models is a CLI flag:  --provider groq --model llama-3.3-70b-versatile
Any OpenAI-compatible endpoint (incl. one Kuku provides) works via:
    --provider custom --base-url https://... --api-key-env MY_KEY --model my-model
Responses are cached on disk, so re-runs are free and deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PRESETS = {
    # name: (base_url, api_key_env, default_model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3.6-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    "ollama": ("http://localhost:11434/v1", "", "llama3.1:8b"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
}


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """Daily/plan quota hit: retrying won't help today, switch model instead."""


class LLM:
    def __init__(self, provider: str, model: str | None = None, base_url: str | None = None,
                 api_key_env: str | None = None, cache_dir: str = ".cache", temperature: float = 0.0,
                 min_interval: float = 0.0, reasoning_effort: str | None = "auto"):
        if provider not in PRESETS and provider != "custom":
            raise LLMError(f"unknown provider {provider!r}; choose from {list(PRESETS)} or 'custom'")
        base, env, default_model = PRESETS.get(provider, ("", "", ""))
        self.provider = provider
        self.base_url = (base_url or base).rstrip("/")
        self.key_env = api_key_env if api_key_env is not None else env
        self.api_key = os.environ.get(self.key_env, "") if self.key_env else ""
        # "--model a,b,c" = fallback chain: free tiers cap requests per model per day
        # (Gemini: 20/day/model), so a 50-episode show can spill over to the next model.
        self.models = [m.strip() for m in (model or default_model).split(",") if m.strip()]
        self.model = self.models[0]
        self.last_model = self.model
        self.temperature = temperature
        self.min_interval = min_interval  # free tiers rate-limit; space calls out
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        # Thinking models spend output budget on hidden reasoning; scoring clips doesn't need much.
        if reasoning_effort == "auto":
            reasoning_effort = "low" if provider == "gemini" else None
        self.reasoning_effort = reasoning_effort
        self.usage = {"calls": 0, "cached": 0, "in_tokens": 0, "out_tokens": 0}
        self._last = 0.0
        if not self.base_url:
            raise LLMError("custom provider needs --base-url")
        if self.key_env and not self.api_key:
            raise LLMError(f"set {self.key_env} in .env (see .env.example) or your environment to use provider {provider!r}")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    self._last = time.time()
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                self._last = time.time()
                full = e.read().decode(errors="replace")
                msg = full[:400]
                if e.code == 429 and re.search(r"PerDay|per day|RESOURCE_EXHAUSTED.*quota|insufficient_quota", full, re.S):
                    raise QuotaExhausted(f"daily quota exhausted for {self.name}") from None
                if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                    pause = min(60, 4 * 2 ** attempt)
                    print(f"  [{self.provider}] HTTP {e.code}, retry {attempt + 1}/5 in {pause}s: {msg[:120]!r}", file=sys.stderr)
                    time.sleep(pause)
                    continue
                raise LLMError(f"HTTP {e.code} from {self.provider}: {msg}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  [{self.provider}] network/timeout ({e}), retry {attempt + 1}/5", file=sys.stderr)
                if attempt < 5:
                    time.sleep(3 * 2 ** attempt)
                    continue
                raise LLMError(f"cannot reach {url}: {e}") from None
        raise LLMError("retries exhausted")

    def _call(self, system: str, user: str, max_tokens: int) -> tuple[str, str, int, int]:
        if self.provider == "anthropic":
            r = self._post(f"{self.base_url}/messages",
                           {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature,
                            "system": system, "messages": [{"role": "user", "content": user}]},
                           {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
            text = "".join(b.get("text", "") for b in r.get("content", []))
            u = r.get("usage", {})
            finish = "length" if r.get("stop_reason") == "max_tokens" else "stop"
            return text, finish, u.get("input_tokens", 0), u.get("output_tokens", 0)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        try:
            r = self._post(f"{self.base_url}/chat/completions", body, headers)
        except LLMError as e:
            if self.reasoning_effort and "reasoning" in str(e).lower():
                self.reasoning_effort = None  # endpoint doesn't support it; carry on without
                return self._call(system, user, max_tokens)
            raise
        try:
            ch = r["choices"][0]
            text, finish = ch["message"].get("content") or "", ch.get("finish_reason") or "stop"
        except (KeyError, IndexError):
            raise LLMError(f"unexpected response: {str(r)[:300]}") from None
        u = r.get("usage", {}) or {}
        return text, finish, u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0

    def _key(self, model: str, system: str, user: str) -> Path:
        name = f"{self.provider}:{model}"
        return self.cache / (hashlib.sha256(json.dumps([name, self.temperature, system, user]).encode()).hexdigest()[:24] + ".json")

    def complete_json(self, system: str, user: str, max_tokens: int = 8000):
        """Cached JSON call. Only answers that parse are cached, so a bad/truncated reply
        is retried next run instead of being replayed forever. On a daily-quota error the
        next model in the fallback chain takes over; self.last_model records who answered."""
        for m in self.models:  # reuse any model's cached answer
            cpath = self._key(m, system, user)
            if cpath.exists():
                try:
                    data = extract_json(json.loads(cpath.read_text())["text"])
                    self.usage["cached"] += 1
                    self.last_model = m
                    return data
                except (LLMError, KeyError, json.JSONDecodeError):
                    cpath.unlink()
        while True:
            try:
                return self._fresh(system, user, max_tokens)
            except QuotaExhausted as e:
                i = self.models.index(self.model)
                if i + 1 >= len(self.models):
                    raise
                print(f"  [{self.provider}] {e}; switching to {self.models[i + 1]}", file=sys.stderr)
                self.model = self.models[i + 1]

    def _fresh(self, system: str, user: str, max_tokens: int):
        cpath = self._key(self.model, system, user)
        last_err = None
        for budget in (max_tokens, max_tokens * 2):
            text, finish, tin, tout = self._call(system, user, budget)
            self.usage["calls"] += 1
            self.usage["in_tokens"] += tin
            self.usage["out_tokens"] += tout
            try:
                data = extract_json(text)
            except LLMError as e:
                last_err = LLMError(f"{e} (finish_reason={finish})")
                continue  # retry once with double the budget
            cpath.write_text(json.dumps({"model": self.name, "text": text, "usage": [tin, tout]}))
            self.last_model = self.model
            return data
        raise last_err


def extract_json(text: str):
    """Models wrap JSON in prose or ``` fences; take the outermost object/array."""
    text = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model did not return valid JSON: {text[:200]}")

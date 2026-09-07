"""
Resilient OpenRouter API Client.
Supports high-concurrency model invocations, JSON schema enforcement,
live pricing sync, rate limiting, and exponential backoff retry logic.
"""

import asyncio
import json
import math
import os
import random
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from config import (
    OPENROUTER_COMPLETIONS_URL,
    OPENROUTER_MODELS_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_INITIAL_RPS,
    DEFAULT_MIN_RPS,
    DEFAULT_MAX_RPS
)
from core.schemas import EMAIL_ANALYSIS_SCHEMA, repair_json_content, validate_email_analysis

# httpx is preferred when available; urllib is the dependency-free fallback.
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False



def _sanitize_provider_error_detail(detail: Any, *, limit: int = 300) -> str:
    """Truncate provider error bodies for logs (Chunk 12A).

    Full provider payloads can contain request-specific noise; logs keep a
    short, single-line summary instead of dumping bodies verbatim.
    """
    if detail is None:
        return ""
    if isinstance(detail, (dict, list)):
        try:
            text = json.dumps(detail, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(detail)
    else:
        text = str(detail)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text

class NonRetryableAPIError(RuntimeError):
    """An API error that cannot be repaired by retrying the same request."""


class RateLimitExhaustedError(RuntimeError):
    """All configured 429 attempts were exhausted without a further retry."""


def fetch_openrouter_pricing(api_key: str) -> Dict[str, Dict[str, float]]:
    """
    Fetches real-time pricing (prompt & completion cost per 1k tokens)
    from OpenRouter's GET /models endpoint.
    """
    headers = {
        "User-Agent": "OpenRouterBenchmark/3.0"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    pricing_map = {}
    try:
        req = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_list = data.get("data", [])
            for m in models_list:
                model_id = m.get("id")
                pricing = m.get("pricing", {})
                
                # Prices in OpenRouter API are per 1 token (or string representation)
                try:
                    raw_prompt = pricing.get("prompt")
                    raw_completion = pricing.get("completion")
                    p_prompt = float(raw_prompt) * 1000.0 if raw_prompt is not None else None
                    p_completion = float(raw_completion) * 1000.0 if raw_completion is not None else None
                    if p_prompt is not None and (not math.isfinite(p_prompt) or p_prompt < 0):
                        p_prompt = None
                    if p_completion is not None and (not math.isfinite(p_completion) or p_completion < 0):
                        p_completion = None
                except (ValueError, TypeError):
                    p_prompt, p_completion = None, None

                context_len = m.get("context_length", 0)
                pricing_map[model_id] = {
                    "prompt_price_per_1k": p_prompt,
                    "completion_price_per_1k": p_completion,
                    "context_length": context_len
                }
    except Exception as e:
        print(f"Warning: Failed to fetch live OpenRouter pricing: {e}", file=sys.stderr)

    return pricing_map


class RateLimiter:
    """Adaptive per-model request-spacing limiter with fractional RPS support."""

    def __init__(
        self,
        initial_rps: float = DEFAULT_INITIAL_RPS,
        min_rps: float = DEFAULT_MIN_RPS,
        max_rps: float = DEFAULT_MAX_RPS,
    ):
        if initial_rps <= 0 or min_rps <= 0 or max_rps <= 0:
            raise ValueError("RPS values must be greater than zero")
        if min_rps > max_rps:
            raise ValueError("min_rps cannot exceed max_rps")
        self.rps = min(max(initial_rps, min_rps), max_rps)
        self.min_rps = min_rps
        self.max_rps = max_rps
        self._window_s = 1.0
        self._timestamps: list = []
        self._consecutive_success = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Reserve a request slot without holding the mutex while sleeping."""
        while True:
            async with self._lock:
                now = time.perf_counter()
                interval_s = 1.0 / self.rps
                latest = self._timestamps[-1] if self._timestamps else None
                if latest is None or now - latest >= interval_s:
                    self._timestamps[:] = [now]
                    return
                wait_s = max(0.001, interval_s - (now - latest))
            await asyncio.sleep(wait_s)

    def report_429(self):
        old_rps = self.rps
        self.rps = max(self.min_rps, self.rps / 2.0)
        self._consecutive_success = 0
        if self.rps != old_rps:
            print(
                f"[RateLimiter] 429 received, backing off {old_rps:.2f} -> {self.rps:.2f} req/s",
                file=sys.stderr,
            )

    def report_success(self):
        self._consecutive_success += 1
        if self._consecutive_success >= 20 and self.rps < self.max_rps:
            self.rps = min(self.max_rps, self.rps * 1.2)
            self._consecutive_success = 0


def compute_call_cost(model_id, prompt_tokens, completion_tokens, pricing_map):
    """Compute the USD cost of a call using the exact model/route price only.

    Returns ``None`` when the price for this exact model_id is unknown. Callers
    must treat ``None`` as "unknown cost", never coerce it to 0.0 — an unknown
    price is not the same as a free call. There is deliberately no fallback to
    a "base model" price for provider variants (e.g. ``vendor/model:variant``),
    since variant pricing can differ from the base route.
    """
    p_info = pricing_map.get(model_id)
    if not isinstance(p_info, dict):
        return None
    prompt_price = p_info.get("prompt_price_per_1k")
    completion_price = p_info.get("completion_price_per_1k")
    if not isinstance(prompt_price, (int, float)) or isinstance(prompt_price, bool):
        return None
    if not isinstance(completion_price, (int, float)) or isinstance(completion_price, bool):
        return None
    if not math.isfinite(float(prompt_price)) or not math.isfinite(float(completion_price)):
        return None
    if float(prompt_price) < 0 or float(completion_price) < 0:
        return None
    prompt_cost = (prompt_tokens / 1000.0) * float(prompt_price)
    completion_cost = (completion_tokens / 1000.0) * float(completion_price)
    return prompt_cost + completion_cost


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, pricing_map: Optional[Dict[str, Dict[str, float]]] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)

        # An explicitly supplied pricing_map (including an empty dict) must be
        # honored as-is. Only the true "not supplied" case (None) triggers a
        # live fetch — `x or fetch(...)` would incorrectly treat an explicit
        # {} the same as "not supplied" and always hit the network.
        self.pricing_map = fetch_openrouter_pricing(self.api_key) if pricing_map is None else pricing_map

        # One RateLimiter per model, created lazily — different models on
        # OpenRouter have different provider-side limits, so they shouldn't
        # share a single throttle.
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._http_clients: Dict[float, Any] = {}

    async def _get_http_client(self, timeout: int):
        if not HAS_HTTPX:
            return None
        timeout_key = float(timeout)
        client = self._http_clients.get(timeout_key)
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
            self._http_clients[timeout_key] = client
        return client

    async def aclose(self):
        clients = list(self._http_clients.values())
        self._http_clients.clear()
        for client in clients:
            await client.aclose()

    def _get_rate_limiter(self, model: str) -> RateLimiter:
        rl = self._rate_limiters.get(model)
        if rl is None:
            rl = RateLimiter()
            self._rate_limiters[model] = rl
        return rl

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[float]:
        """Return Retry-After as non-negative seconds for delta-seconds or HTTP-date."""
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

    async def call_model_async(
        self,
        model: str,
        messages: list,
        response_format: Optional[dict] = None,
        temperature: float = 0.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_tokens: Optional[int] = None,
        max_429_retries: Optional[int] = None,
        max_transport_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a request with independent retry budgets.

        ``latency_ms`` is the wall-clock duration of the successful provider
        HTTP request only; client-side rate-limit waits and retry backoff are
        intentionally excluded.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openrouter-benchmark",
            "X-Title": "Mass-Email-Benchmark-Engine-v3"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        if response_format:
            payload["response_format"] = response_format
        # FIX (Chunk 1, finding #18): max_tokens was never sent, leaving
        # output truncation entirely up to model/provider defaults -- which
        # can change outside the benchmark's control and isn't reproducible.
        # An explicit None still means "no request-side limit" (caller opted
        # out), but every caller in this codebase now passes a real value.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        backoff_delay = 1.0
        max_backoff_s = 15.0
        max_429_attempts = max(1, int(max_429_retries if max_429_retries is not None else max_retries))
        max_transport_attempts = max(1, int(
            max_transport_retries if max_transport_retries is not None else max_retries
        ))
        rate_limit_attempts = 0
        transport_attempts = 0
        rate_limiter = self._get_rate_limiter(model)

        while True:
            await rate_limiter.acquire()
            try:
                if HAS_HTTPX:
                    client = await self._get_http_client(timeout)
                    request_start = time.perf_counter()
                    resp = await client.post(
                        OPENROUTER_COMPLETIONS_URL, headers=headers, json=payload
                    )
                    request_elapsed_ms = (time.perf_counter() - request_start) * 1000.0
                    status_code = resp.status_code
                    response_headers = getattr(resp, "headers", {})
                    if status_code == 429:
                        retry_after = response_headers.get("retry-after")
                        data = None
                    else:
                        if 400 <= status_code < 500:
                            try:
                                detail = resp.json()
                            except Exception:
                                detail = _sanitize_provider_error_detail(resp.text)
                            raise NonRetryableAPIError(
                                f"OpenRouter returned HTTP {status_code} for model '{model}': {_sanitize_provider_error_detail(detail)}"
                            )
                        if status_code >= 500:
                            resp.raise_for_status()
                        data = resp.json()
                else:
                    loop = asyncio.get_event_loop()

                    def _sync_post():
                        req = urllib.request.Request(
                            OPENROUTER_COMPLETIONS_URL,
                            data=json.dumps(payload).encode("utf-8"),
                            headers=headers,
                        )
                        request_start = time.perf_counter()
                        try:
                            with urllib.request.urlopen(req, timeout=timeout) as response:
                                body = response.read().decode("utf-8")
                                return (
                                    response.getcode(),
                                    response.headers,
                                    json.loads(body),
                                    (time.perf_counter() - request_start) * 1000.0,
                                )
                        except urllib.error.HTTPError as exc:
                            if exc.code == 429:
                                return (
                                    429,
                                    exc.headers,
                                    None,
                                    (time.perf_counter() - request_start) * 1000.0,
                                )
                            body = exc.read().decode("utf-8", errors="replace")[:2000]
                            if 400 <= exc.code < 500:
                                raise NonRetryableAPIError(
                                    f"OpenRouter returned HTTP {exc.code} for model '{model}': {body}"
                                ) from exc
                            raise

                    status_code, response_headers, data, request_elapsed_ms = await loop.run_in_executor(
                        None, _sync_post
                    )
                    retry_after = response_headers.get("retry-after") if response_headers else None

                if status_code == 429:
                    rate_limit_attempts += 1
                    rate_limiter.report_429()
                    if rate_limit_attempts >= max_429_attempts:
                        raise RateLimitExhaustedError(
                            f"OpenRouter API call for model '{model}' failed: rate limited "
                            f"(HTTP 429) after {rate_limit_attempts} attempts."
                        )
                    wait_s = self._parse_retry_after(retry_after)
                    if wait_s is None:
                        wait_s = min(backoff_delay, max_backoff_s)
                    wait_s += random.uniform(0.1, 0.5)
                    print(
                        f"[api_client] {model}: rate limited (429), "
                        f"retry {rate_limit_attempts}/{max_429_attempts} in {wait_s:.1f}s",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait_s)
                    backoff_delay = min(backoff_delay * 2.0, max_backoff_s)
                    continue

                if "choices" not in data or not data.get("choices"):
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' is missing 'choices': {_sanitize_provider_error_detail(data)}"
                    )
                if "usage" not in data:
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' is missing 'usage': {_sanitize_provider_error_detail(data)}"
                    )

                choice = data["choices"][0]
                if not isinstance(choice, dict):
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has invalid first choice: {choice!r}"
                    )
                if choice.get("error") is not None or data.get("error") is not None:
                    detail = choice.get("error") or data.get("error")
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' contains a provider error: "
                        f"{_sanitize_provider_error_detail(detail)}"
                    )
                finish_reason = choice.get("finish_reason")
                if finish_reason in {"error", "failed"}:
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has unsuccessful finish_reason: "
                        f"{finish_reason!r}"
                    )
                message_obj = choice.get("message")
                if not isinstance(message_obj, dict):
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has invalid choice.message: {message_obj!r}"
                    )
                if "content" not in message_obj or not isinstance(message_obj.get("content"), str):
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has invalid message.content: {message_obj.get('content')!r}"
                    )
                content = message_obj["content"]
                usage = data["usage"]
                if not isinstance(usage, dict):
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has invalid 'usage': {usage!r}"
                    )
                if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has incomplete 'usage': {usage!r}"
                    )
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                if (
                    isinstance(prompt_tokens, bool) or isinstance(completion_tokens, bool)
                    or not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int)
                    or prompt_tokens < 0 or completion_tokens < 0
                ):
                    raise NonRetryableAPIError(
                        f"OpenRouter response for model '{model}' has invalid token counts: {usage!r}"
                    )
                rate_limiter.report_success()
                cost_usd = compute_call_cost(model, prompt_tokens, completion_tokens, self.pricing_map)
                provider_model_id = data.get("model")
                system_fingerprint = data.get("system_fingerprint")

                return {
                    "content": content,
                    "latency_ms": request_elapsed_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost_usd,
                    "raw_response": data,
                    "model": model,
                    "provider_model_id": provider_model_id,
                    "system_fingerprint": system_fingerprint,
                }

            except (NonRetryableAPIError, RateLimitExhaustedError):
                raise
            except Exception as e:
                transport_attempts += 1
                if transport_attempts >= max_transport_attempts:
                    raise RuntimeError(
                        f"OpenRouter API call failed after {transport_attempts} transport attempts: {e}"
                    ) from e
                jitter = random.uniform(0.1, 0.5)
                wait_s = min(backoff_delay, max_backoff_s) + jitter
                print(
                    f"[api_client] {model}: transport error on attempt {transport_attempts}/"
                    f"{max_transport_attempts} ({e}), retrying in {wait_s:.1f}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait_s)
                backoff_delay = min(backoff_delay * 2.0, max_backoff_s)

    async def analyze_email_async(
        self, model: str, system_prompt: str, email_text: str, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sends email for structured analysis using json_schema mode, returning parsed dict & metadata.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "email_analysis",
                "strict": True,
                "schema": EMAIL_ANALYSIS_SCHEMA
            }
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Email text to analyze:\n\n{email_text}"}
        ]

        res = await self.call_model_async(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        content = res["content"]
        parsed = None
        is_valid_schema = False
        # FIX (finding #30): a repaired parse must be distinguishable from a
        # clean one. Silently swapping in `repair_json_content()`'s output
        # let "the model returned valid JSON" and "the raw output was
        # malformed and we patched around it" look identical downstream, so
        # a benchmark consumer could not tell a genuinely well-formed
        # response from a repaired one even though both currently exist
        # under the same is_valid_schema flag. Record the original parse
        # failure and whether repair was used, unconditionally -- these are
        # independent of whatever is_valid_schema ends up being.
        raw_parse_error: Optional[str] = None
        parse_repaired = False

        # Attempt JSON parsing & schema validation
        try:
            parsed = json.loads(content)
            is_valid, _ = validate_email_analysis(parsed)
            is_valid_schema = is_valid
        except Exception as exc:
            raw_parse_error = str(exc)
            # Fallback JSON repair
            repaired = repair_json_content(content)
            if repaired:
                parsed = repaired
                parse_repaired = True
                is_valid, _ = validate_email_analysis(repaired)
                is_valid_schema = is_valid

        if not parsed:
            parsed = {}

        return {
            **res,
            "parsed": parsed,
            "is_valid_schema": is_valid_schema,
            "parse_repaired": parse_repaired,
            "raw_parse_error": raw_parse_error,
        }

    async def elaborate_email_async(
        self, model: str, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sends request to elaborate/write an email.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return await self.call_model_async(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )
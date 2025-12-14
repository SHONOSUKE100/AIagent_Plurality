"""Helpers for handling OpenAI rate limits with header-aware backoff."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Type

from camel.models import BaseModelBackend
from camel.messages import OpenAIMessage
from camel.types import ChatCompletion, ChatCompletionChunk
from openai import AsyncStream, Stream

# Optional OpenAI imports – keep the wrapper usable even if the SDK is absent.
try:  # pragma: no cover - import guard for optional dependency
    from openai import APIStatusError, RateLimitError  # type: ignore
except Exception:  # pragma: no cover - fallback when OpenAI is not installed
    APIStatusError = None
    RateLimitError = tuple()  # type: ignore[misc]

LOG_PATH = Path("log/ratelimit.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("ratelimit")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class RateLimitAwareModelWrapper(BaseModelBackend):
    """Wrap a CAMEL BaseModelBackend to retry on rate limits using headers."""

    def __init__(
        self,
        backend: BaseModelBackend,
        *,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
        jitter_ratio: float = 0.25,
    ) -> None:
        # Keep backend available before BaseModelBackend.__init__ triggers check_model_config.
        self._backend = backend

        # Initialize BaseModelBackend with the underlying model config
        super().__init__(
            model_type=backend.model_type,
            model_config_dict=getattr(backend, "model_config_dict", None),
            api_key=getattr(backend, "_api_key", None),
            url=getattr(backend, "_url", None),
            token_counter=getattr(backend, "_token_counter", None),
            timeout=getattr(backend, "_timeout", None),
        )
        self.max_retries = max(0, max_retries)
        self.base_delay = max(base_delay, 0.1)
        self.max_delay = max(max_delay, self.base_delay)
        self.jitter_ratio = max(jitter_ratio, 0.0)
        self._logger = _get_logger()

    # BaseModelBackend requires a token_counter property
    @property
    def token_counter(self):
        return self._backend.token_counter

    def check_model_config(self) -> None:
        """Defer model config validation to the wrapped backend."""
        try:
            self._backend.check_model_config()
        except AttributeError:
            # Some backends may not implement this explicitly.
            pass

    # ---- Public API passthrough with retry wrappers ---------------------------------
    def __call__(self, *args, **kwargs):
        return self._call_with_retry(self._backend, *args, **kwargs)

    def run(
        self,
        messages: list[OpenAIMessage],
        response_format: Optional[Type] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        return self._call_with_retry(
            self._backend.run,
            messages=messages,
            response_format=response_format,
            tools=tools,
        )

    async def arun(
        self,
        messages: list[OpenAIMessage],
        response_format: Optional[Type] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        return await self._call_with_retry_async(
            self._backend.arun,
            messages=messages,
            response_format=response_format,
            tools=tools,
        )

    # Delegate low-level calls required by BaseModelBackend
    def _run(
        self,
        messages: list[OpenAIMessage],
        response_format: Optional[Type] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        return self.run(messages, response_format=response_format, tools=tools)

    async def _arun(
        self,
        messages: list[OpenAIMessage],
        response_format: Optional[Type] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        return await self.arun(messages, response_format=response_format, tools=tools)

    def __getattr__(self, name: str):
        # Delegate everything else to the underlying model. Avoid infinite recursion.
        if name == "_backend":
            raise AttributeError(name)
        return getattr(self._backend, name)

    # ---- Retry helpers ---------------------------------------------------------------
    def _call_with_retry(self, func: Callable, *args, **kwargs):
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                attempt += 1
                if not self._should_retry(exc, attempt):
                    raise
                delay = self._compute_delay(exc, attempt)
                headers = self._extract_headers(exc)
                self._log_backoff(exc, attempt, delay, headers)
                time.sleep(delay)

    async def _call_with_retry_async(self, func: Callable[..., Awaitable], *args, **kwargs):
        attempt = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                attempt += 1
                if not self._should_retry(exc, attempt):
                    raise
                delay = self._compute_delay(exc, attempt)
                headers = self._extract_headers(exc)
                self._log_backoff(exc, attempt, delay, headers)
                await asyncio.sleep(delay)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt > self.max_retries:
            return False

        if RateLimitError and isinstance(exc, RateLimitError):
            return True
        if APIStatusError and isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            return status in {429, 500, 503}

        status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {429, 500, 503}:
            return True

        return False

    def _compute_delay(self, exc: Exception, attempt: int) -> float:
        headers = self._extract_headers(exc)
        delay_candidates: list[float] = []

        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                delay_candidates.append(float(retry_after))
            except (TypeError, ValueError):
                pass

        reset_at = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
        if reset_at:
            try:
                reset_time = float(reset_at)
                remaining = max(0.0, reset_time - time.time())
                delay_candidates.append(remaining)
            except (TypeError, ValueError, OSError):
                pass

        exponential = self.base_delay * (2 ** (attempt - 1))
        delay_candidates.append(exponential)

        delay = min(max(delay_candidates), self.max_delay)
        jitter = 1 + (random.random() * self.jitter_ratio)
        return max(self.base_delay, min(delay * jitter, self.max_delay))

    def _extract_headers(self, exc: Exception) -> Mapping[str, Any]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return {}
        if isinstance(headers, Mapping):
            return headers
        try:
            return dict(headers)
        except Exception:
            return {}

    def _log_backoff(
        self,
        exc: Exception,
        attempt: int,
        delay: float,
        headers: Mapping[str, Any],
    ) -> None:
        remaining = headers.get("x-ratelimit-remaining-requests") or headers.get("X-RateLimit-Remaining-Requests")
        reset = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
        message = (
            f"Rate limit/backoff: attempt {attempt}/{self.max_retries}, "
            f"sleep {delay:.2f}s, remaining={remaining}, reset={reset}"
        )
        self._logger.warning(message)
        self._logger.debug("Headers seen during rate limit: %s", headers)

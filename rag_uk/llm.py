from __future__ import annotations

import time
from dataclasses import dataclass
from time import perf_counter

import httpx

from .calllog import CallLog
from .config import LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float


class LLMClient:
    def __init__(self, cfg: LLMConfig, call_log: CallLog | None = None):
        self.cfg = cfg
        self.call_log = call_log

    def _log(self, **kwargs) -> None:
        if self.call_log:
            self.call_log.record(kind="chat", model=self.cfg.model or "", **kwargs)

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        if not self.cfg.configured:
            raise LLMError("Не задано LLM_API_BASE або LLM_MODEL")

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }

        last_error = "невідома помилка"
        with httpx.Client(timeout=self.cfg.timeout_s) as client:
            for attempt in range(self.cfg.max_retries + 1):
                started = perf_counter()
                try:
                    response = client.post(url, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    latency = perf_counter() - started
                    last_error = f"{type(exc).__name__}: {exc}"
                    self._log(
                        input_tokens=None,
                        output_tokens=None,
                        latency_s=latency,
                        ok=False,
                        error=last_error,
                    )
                    if attempt < self.cfg.max_retries:
                        time.sleep(self._delay(attempt))
                        continue
                    raise LLMError(last_error) from exc

                latency = perf_counter() - started
                status = response.status_code
                if status != 200:
                    last_error = f"http {status}: {response.text[:200]}"
                    self._log(
                        input_tokens=None,
                        output_tokens=None,
                        latency_s=latency,
                        ok=False,
                        error=last_error,
                    )
                    retryable = status == 429 or status >= 500
                    if retryable and attempt < self.cfg.max_retries:
                        time.sleep(self._delay(attempt))
                        continue
                    raise LLMError(last_error)

                data = response.json()
                try:
                    text = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    last_error = "відповідь без choices[0].message.content"
                    self._log(
                        input_tokens=None,
                        output_tokens=None,
                        latency_s=latency,
                        ok=False,
                        error=last_error,
                    )
                    raise LLMError(last_error) from exc

                usage = data.get("usage") or {}
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                self._log(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_s=latency,
                )
                return ChatResult(text, self.cfg.model, input_tokens, output_tokens, latency)

        raise LLMError(last_error)

    def _delay(self, attempt: int) -> float:
        return min(
            self.cfg.retry_base_delay_s * (2**attempt), self.cfg.retry_max_delay_s
        )

"""Provider-agnostic LLM client.

Cohere, OpenRouter, OpenAI and Groq all expose an OpenAI-compatible chat
endpoint, so one small wrapper around the `openai` SDK covers them all. The API
key is supplied by the user in the app at run time and is never written to disk.
"""

from __future__ import annotations

from . import config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str | None = None):
        from openai import OpenAI

        if provider not in config.PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}")
        if not api_key or not api_key.strip():
            raise ValueError("An API key is required.")
        settings = config.PROVIDERS[provider]
        self.provider = provider
        self.model = (model or settings["default_model"]).strip()
        self._client = OpenAI(
            api_key=api_key.strip(),
            base_url=settings["base_url"],
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=2,  # SDK retries transient network / rate-limit errors with backoff
        )

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: list[dict], json_mode: bool = False, max_tokens: int | None = None) -> str:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        )
        try:
            if json_mode:
                try:
                    resp = self._client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except Exception as exc:  # some models do not support JSON mode -> prompt-only JSON
                    text = str(exc).lower()
                    if "response_format" in text or "json" in text:
                        resp = self._client.chat.completions.create(**kwargs)
                    else:
                        raise
            else:
                resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(friendly_error(exc)) from exc
        return (resp.choices[0].message.content or "").strip()

    def test_connection(self) -> tuple[bool, str]:
        try:
            reply = self.chat([{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
                              json_mode=True, max_tokens=20)
        except LLMError as exc:
            return False, str(exc)
        return True, f"Connected to {self.label}" + ("" if "ok" in reply.lower() else " (unexpected reply, but the key works)")


def friendly_error(exc: Exception) -> str:
    """Turn SDK exceptions into messages a non-technical user can act on."""
    text = str(exc)
    low = text.lower()
    if "401" in text or "unauthorized" in low or "invalid api key" in low or "invalid_api_key" in low:
        return "The API key was rejected (401). Check that you pasted the full key for the selected provider."
    if "402" in text or "insufficient" in low or "credit" in low:
        return "The provider says the account has no credit left (402). Add credit or switch provider."
    if "429" in text or "rate limit" in low or "too many requests" in low:
        return "Rate limit reached (429). Wait a minute and try again - trial keys allow only a few calls per minute."
    if "404" in text or "model" in low and "not found" in low:
        return "The model name was not found (404). Use the default model for this provider."
    if "timeout" in low or "timed out" in low:
        return "The AI provider took too long to answer. Try again."
    if "connection" in low:
        return "Could not reach the AI provider. Check your internet connection."
    return f"AI provider error: {text[:300]}"

from __future__ import annotations

import httpx
from openai import OpenAI

from rag_learning import config


def _is_local(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


class ChatLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ):
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        resolved_base_url = base_url or config.OPENAI_BASE_URL

        # When talking to a local server (e.g. Ollama on localhost), bypass any
        # system proxy set by a VPN. httpx honours HTTP(S)_PROXY by default, which
        # routes localhost requests through the proxy and makes it return 502.
        http_client = None
        if _is_local(resolved_base_url):
            http_client = httpx.Client(trust_env=False)

        self.client = OpenAI(
            api_key=api_key or config.require_api_key(),
            base_url=resolved_base_url,
            http_client=http_client,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            # Factual QA wants the model's most confident answer; a positive
            # frequency_penalty pushes the model away from tokens it has
            # already committed to, which on llama3:8b shows up as
            # yes/no over-triggering. Default of 0 matches the OpenAI default.
            frequency_penalty=0.0,
        )
        return response.choices[0].message.content.strip()

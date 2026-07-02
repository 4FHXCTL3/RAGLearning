from __future__ import annotations

from openai import OpenAI

from rag_learning import config


class ChatLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key or config.require_api_key(),
            base_url=base_url or config.OPENAI_BASE_URL,
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=0.5,
        )
        return response.choices[0].message.content.strip()

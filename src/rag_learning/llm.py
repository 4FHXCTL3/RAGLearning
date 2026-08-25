from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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

    def _do_call(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
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

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        return self._do_call(prompt, system_prompt, temperature)

    def generate_batch(
        self,
        prompts: list[str],
        system_prompts: list[str | None] | None = None,
        temperatures: list[float | None] | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        """Send N independent chat calls in parallel, return responses in order.

        The OpenAI client is thread-safe (httpx serialises concurrent calls
        internally), so one client is shared across worker threads. Default
        max_workers=8 is a safe local-Ollama setting; raise it on a beefy GPU.

        For ``len(prompts) == 1`` we fall back to the serial path to avoid
        spinning up a thread pool for a single call.
        """
        n = len(prompts)
        if n == 0:
            return []
        if n == 1:
            return [self._do_call(
                prompts[0],
                system_prompts[0] if system_prompts else None,
                temperatures[0] if temperatures else None,
            )]
        if system_prompts is None:
            system_prompts = [None] * n
        if temperatures is None:
            temperatures = [None] * n
        workers = max_workers or min(8, n)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(self._do_call, p, sp, t)
                for p, sp, t in zip(prompts, system_prompts, temperatures)
            ]
            return [f.result() for f in futures]

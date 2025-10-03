import os
from typing import List

import httpx
import openai

from evalplus.gen.util import openai_request
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import concurrent_call


class OpenAIChatDecoder(DecoderBase):
    def __init__(
        self, name: str, base_url=None, verify_certificate=True, **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.verify_certificate = verify_certificate

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)
        prompt = self.instruction_prefix + f"\n```python\n{prompt.strip()}\n```"

        # use concurrency based batching for o1 and deepseek models
        if self.name.startswith("o1-") or self.name == "deepseek-chat":
            return self._codegen_batch_via_concurrency(prompt, num_samples)

        return self._codegen_api_batch(prompt, batch_size)

    def _codegen_api_batch(self, prompt: str, batch_size: int) -> List[str]:
        import json

        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"),
            base_url=self.base_url,
            http_client=httpx.Client(verify=self.verify_certificate),
        )

        # Parse user-provided extra_body and extra_headers
        extra_body = json.loads(self.extra_body) if self.extra_body else {}
        extra_headers = json.loads(self.extra_headers) if self.extra_headers else {}

        # Prepare additional parameters for OpenAI-compatible servers
        extra_params = {}

        # Only add parameters that are actually set (not None)
        if self.top_p is not None:
            extra_params['top_p'] = self.top_p
        if self.top_k is not None:
            extra_body['top_k'] = self.top_k
        if self.presence_penalty is not None:
            extra_params['presence_penalty'] = self.presence_penalty
        if self.repetition_penalty is not None:
            extra_body['repetition_penalty'] = self.repetition_penalty
        if self.max_output_tokens is not None:
            extra_params['max_completion_tokens'] = self.max_output_tokens

        # Add extra_body and extra_headers if they have content
        if extra_body:
            extra_params['extra_body'] = extra_body
        if extra_headers:
            extra_params['extra_headers'] = extra_headers

        ret = openai_request.make_auto_request(
            client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
            **extra_params,
        )

        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)

        return outputs

    def _codegen_batch_via_concurrency(self, prompt: str, batch_size: int) -> List[str]:
        batches = concurrent_call(
            batch_size, self._codegen_api_batch, prompt, batch_size=1
        )
        return [b[0] for b in batches]

    def is_direct_completion(self) -> bool:
        return False

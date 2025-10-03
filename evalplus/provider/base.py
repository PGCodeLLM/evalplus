from abc import ABC, abstractmethod
from typing import List, Optional

from evalplus.provider.utility import EOS


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 768,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        instruction_prefix: str = None,
        response_prefix: str = None,
        # inference parameters (no defaults - user controlled)
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_output_tokens = max_output_tokens
        self.extra_body = extra_body
        self.extra_headers = extra_headers

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

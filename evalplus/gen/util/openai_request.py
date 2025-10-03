import time
from typing import Any

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    stream: bool = False,
    **kwargs
) -> Any:
    kwargs["top_p"] = kwargs.get("top_p", 0.95)
    kwargs["max_completion_tokens"] = kwargs.get("max_completion_tokens", max_tokens)
    if model.startswith("o1-"):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        temperature = 1.0  # o1 models do not support temperature

    if not stream:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            n=n,
            **kwargs
        )

    # Handle streaming
    stream_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=temperature,
        n=n,
        stream=True,
        **kwargs
    )

    # Accumulate chunks
    accumulated_content = [""] * n

    for chunk in stream_response:
        if chunk.choices:
            for choice in chunk.choices:
                idx = choice.index
                if choice.delta.content:
                    accumulated_content[idx] += choice.delta.content

    # Build response object compatible with non-streaming format
    class StreamedChoice:
        def __init__(self, content, index):
            self.message = type('obj', (object,), {'content': content})()
            self.index = index

    class StreamedResponse:
        def __init__(self, choices):
            self.choices = choices

    choices = [StreamedChoice(content, i) for i, content in enumerate(accumulated_content)]
    return StreamedResponse(choices)


def make_auto_request(*args, **kwargs) -> Any:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret

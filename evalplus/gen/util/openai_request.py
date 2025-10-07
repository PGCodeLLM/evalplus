import time

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    stream: bool = False,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = kwargs.get("top_p", 0.95)
    kwargs["max_completion_tokens"] = kwargs.get("max_completion_tokens", max_tokens)

    if model.startswith("o1-"):
        kwargs.pop("top_p", None)
        kwargs.pop("max_completion_tokens", None)
        temperature = 1.0

    if not stream:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            temperature=temperature,
            n=n,
            **kwargs
        )

    stream_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        temperature=temperature,
        n=n,
        stream=True,
        **kwargs
    )

    accumulated_content = [""] * n
    finish_reasons = [None] * n
    roles = [None] * n

    for chunk in stream_response:
        if not chunk.choices:
            continue
        for choice in chunk.choices:
            idx = choice.index
            if choice.delta:
                if choice.delta.content:
                    accumulated_content[idx] += choice.delta.content
                if choice.delta.role:
                    roles[idx] = choice.delta.role
            if choice.finish_reason:
                finish_reasons[idx] = choice.finish_reason

    choices = []
    for i in range(n):
        role = roles[i] if roles[i] else "assistant"
        finish_reason = finish_reasons[i] if finish_reasons[i] else "stop"
        choices.append(
            Choice(
                index=i,
                finish_reason=finish_reason,
                message=ChatCompletionMessage(
                    role=role,
                    content=accumulated_content[i],
                ),
            )
        )

    return ChatCompletion(
        id="streamed-" + str(int(time.time())),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=choices,
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
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

"""Explicit raw-token candidate protocol; copied beside the SLLM backend.

This module intentionally does not change placement or serialize requests.
Automatic migration and concurrent calls to the same backend are outside its
preflight admission. The existing backend's InferenceStatus still owns streaming
state. Full serving concurrency requires a separate correctness run.
"""

import time

from generation_policy import prepare

PROTOCOL = "crosspool-preflight-v1"


def validate_request(data):
    from chat_policy import CHAT_PROTOCOL, validate_chat
    if data.get("baseline_protocol") == CHAT_PROTOCOL:
        return validate_chat(data)
    allowed = {"baseline_protocol", "model", "input_ids", "max_tokens", "do_sample",
               "eos_token_id", "pad_token_id", "stream", "request_id"}
    if set(data) - allowed:
        raise ValueError(f"Unsupported protocol fields: {sorted(set(data) - allowed)}")
    if data.get("baseline_protocol") != PROTOCOL or data.get("stream") is not False:
        raise ValueError("Require the explicit nonstreaming preflight protocol")
    if data.get("do_sample") is not False:
        raise ValueError("Only explicit greedy generation is admitted")
    ids = data.get("input_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 4096 or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("input_ids must contain 1..4096 nonnegative integer token IDs")
    maximum = data.get("max_tokens")
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("max_tokens must be within the fixed 1..8 qualification budget")
    eos = data.get("eos_token_id")
    if not isinstance(eos, list) or not eos or any(type(i) is not int or i < 0 for i in eos):
        raise ValueError("Require an explicit nonempty EOS list")
    pad = data.get("pad_token_id")
    if type(pad) is not int or pad < 0:
        raise ValueError("Require an explicit PAD token")
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise ValueError("Require model identity")
    return data


def generate_baseline(backend, data):
    data = dict(data)
    channel = data.pop("_stream_queue", None)
    import torch
    from transformers import GenerationConfig

    try:
        from chat_policy import CHAT_PROTOCOL, render_request
        chat_receipt = None
        if data.get("baseline_protocol") == CHAT_PROTOCOL:
            data, chat_receipt = render_request(backend.tokenizer, data)
        validate_request(data)
        with backend.status_lock:
            if backend.status.name != "RUNNING" or backend.model is None:
                raise ValueError("Model not ready")
        backend.model.config._attn_implementation = "eager"
        config = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=data["max_tokens"],
            eos_token_id=data["eos_token_id"], pad_token_id=data["pad_token_id"],
            repetition_penalty=1.0, no_repeat_ngram_size=0, use_cache=True)
        effective, flags = prepare(backend.model, config, {
            "max_new_tokens": data["max_tokens"], "eos_token_id": data["eos_token_id"],
            "pad_token_id": data["pad_token_id"]})
        input_ids = torch.tensor([data["input_ids"]], dtype=torch.long, device="cuda:0")
        streamer = backend.inf_status
        if channel is not None:
            from token_stream04 import TokenStreamer
            streamer = TokenStreamer(backend.inf_status,
                lambda event: channel.put(event, timeout=3), data["request_id"], data["max_tokens"],
                tokenizer=backend.tokenizer if chat_receipt is not None else None)
        with torch.inference_mode():
            output = backend.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                generation_config=effective, streamer=streamer, **flags)
        completed_ns = time.monotonic_ns()
        new_ids = output[0, input_ids.shape[1]:].tolist()
        if not new_ids:
            raise ValueError("No generated tokens")
        finish = "stop" if new_ids[-1] in data["eos_token_id"] else "length"
        return {
            "id": "chatcmpl-" + data.get("request_id", "preflight"), "object": "chat.completion",
            "model": data["model"], "created": int(time.time()),
            "choices": [{"index": 0, "message": {"role": "assistant", "content":
                backend.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)}, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(data["input_ids"]), "completion_tokens": len(new_ids),
                      "total_tokens": len(data["input_ids"]) + len(new_ids)},
            "baseline": {"protocol": PROTOCOL, "chat": chat_receipt,
                         "stream_text_tail": streamer.final_delta if channel is not None else "", "input_ids": data["input_ids"], "output_ids": new_ids,
                         "attention_backend": backend.model.config._attn_implementation,
                         "generation_completed_monotonic_ns": completed_ns,
                         "explicit_generation_flags": dict(flags), "effective_generation_config": effective.to_dict()},
        }
    except Exception as error:
        return {"error": f"preflight generation failed: {type(error).__name__}: {error}"}
    finally:
        backend.inf_status.delete()

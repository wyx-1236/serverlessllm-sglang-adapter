"""Raw-token boundary and honest cumulative SGLang stream accounting."""

import time

PROTOCOL = "crosspool-preflight-v1"
FIELDS = {"baseline_protocol", "model", "request_id", "input_ids", "max_tokens",
          "do_sample", "eos_token_id", "pad_token_id", "stream"}


def _ids(value, maximum):
    return (isinstance(value, list) and 1 <= len(value) <= maximum
            and all(type(i) is int and i >= 0 for i in value))


def validate_request(data):
    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError("Require exactly the raw-token protocol fields")
    if data["baseline_protocol"] != PROTOCOL or data["stream"] is not False:
        raise ValueError("HTTP owns streaming; backend requires the raw protocol")
    if data["do_sample"] is not False or not _ids(data["input_ids"], 4096):
        raise ValueError("Require raw IDs and explicit greedy generation")
    if type(data["max_tokens"]) is not int or not 1 <= data["max_tokens"] <= 64:
        raise ValueError("Output budget is 1..64 tokens")
    if not _ids(data["eos_token_id"], 32):
        raise ValueError("Require an explicit EOS list")
    if type(data["pad_token_id"]) is not int or data["pad_token_id"] < 0:
        raise ValueError("Require explicit PAD")
    for key in ("model", "request_id"):
        if not isinstance(data[key], str) or not 1 <= len(data[key]) <= 256:
            raise ValueError("Require bounded model and request identity")
    return data


class StreamReceipt:
    """Accept cumulative native outputs; never invent per-token timing."""

    def __init__(self, request):
        self.request = validate_request(dict(request))
        self.output_ids = []
        self.terminal = None
        self.sequence = 0
        self.completed_ns = None
        self.text = ""

    def push(self, chunk, observed_ns):
        if self.terminal is not None:
            raise ValueError("Output after terminal event")
        ids = chunk.get("output_ids")
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Native stream did not expose token IDs")
        if ids[:len(self.output_ids)] != self.output_ids:
            raise ValueError("Native cumulative output rewrote prior tokens")
        if len(ids) > self.request["max_tokens"]:
            raise ValueError("Native output exceeded request budget")
        meta = chunk.get("meta_info", {})
        if meta.get("id", self.request["request_id"]) != self.request["request_id"]:
            raise ValueError("Native request identity mismatch")
        new = ids[len(self.output_ids):]
        self.output_ids = list(ids)
        text_delta = ""
        if isinstance(chunk.get("text"), str):
            if not chunk["text"].startswith(self.text):
                raise ValueError("Native cumulative text rewrote prior content")
            text_delta = chunk["text"][len(self.text):]
            self.text = chunk["text"]
        if meta.get("finish_reason") is not None:
            self.terminal = dict(meta)
            self.completed_ns = observed_ns
        if not new and not text_delta:
            return None
        self.sequence += 1
        return {"type": "token_chunk" if new else "text_chunk", "request_id": self.request["request_id"],
                "sequence": self.sequence, "token_ids": new,
                "text_delta": text_delta,
                "generated_monotonic_ns": observed_ns,
                "timestamp_boundary": "native_engine_stream_observed"}

    def finish(self):
        if self.terminal is None or not self.output_ids:
            raise ValueError("Missing completed native stream")
        prompt = self.terminal.get("prompt_tokens")
        completion = self.terminal.get("completion_tokens")
        if (type(prompt) is not int or type(completion) is not int
                or prompt != len(self.request["input_ids"])
                or completion != len(self.output_ids)):
            raise ValueError("Native usage disagrees with raw IDs")
        reason = self.terminal["finish_reason"]
        if not isinstance(reason, dict) or reason.get("type") not in {"stop", "length"}:
            raise ValueError("Native request did not finish normally")
        return {"id": "chatcmpl-" + self.request["request_id"],
                "object": "chat.completion", "created": int(time.time()),
                "model": self.request["model"], "choices": [{"index": 0,
                "message": {"role": "assistant", "content": self.text},
                "finish_reason": reason["type"]}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                          "total_tokens": prompt + completion},
                "baseline": {"protocol": PROTOCOL, "engine": "sglang",
                    "input_ids": self.request["input_ids"],
                    "output_ids": self.output_ids, "stream_text_tail": "",
                    "generation_completed_monotonic_ns": self.completed_ns,
                    "native_meta_info": self.terminal,
                    "timing_capability": "chunks; per-token TBT unqualified"}}

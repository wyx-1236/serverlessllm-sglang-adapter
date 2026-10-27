"""Explicit cohort settings and a fresh supervised parent admission gate."""

import json
from pathlib import Path
import time

NO_PLUGINS = "__sllm_no_plugins_20261015__"


def engine_arguments(config):
    if type(config["prefix_cache"]) is not bool:
        raise ValueError("prefix_cache must be an explicit boolean")
    for name in ("kv_tokens", "max_running_requests"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("Require positive explicit capacity")
    if config["kv_tokens"] < 4160:
        raise ValueError("KV pool cannot admit the longest planned request")
    if not config["model_path"].startswith("/"):
        raise ValueError("Model configuration must be an existing local path")
    if config["attention_backend"] not in {"triton", "flashinfer", "fa3"}:
        raise ValueError("Unqualified attention backend")
    return dict(model_path=config["model_path"], tokenizer_path=config["model_path"],
        dtype="bfloat16", kv_cache_dtype="auto", tp_size=1, dp_size=1,
        trust_remote_code=False, random_seed=0, context_length=4160,
        disable_radix_cache=not config["prefix_cache"],
        max_total_tokens=config["kv_tokens"],
        max_running_requests=config["max_running_requests"],
        mem_fraction_static=0.6, chunked_prefill_size=1024,
        attention_backend=config["attention_backend"],
        disable_prefill_cuda_graph=True, disable_decode_cuda_graph=True,
        startup_weight_load_mode="serial", weight_loader_prefetch_checkpoints=False,
        incremental_streaming_output=False, stream_interval=1,
        enable_metrics=False, log_level="warning")


def verify_parent(config, now=None):
    """Read the supervised remote mirror, never create or extend authorization."""
    now = time.time() if now is None else now
    parent = json.loads(Path(config["parent_path"]).read_text())
    unit = config["unit"].removesuffix(".service")
    registered = {item.removesuffix(".service") for item in parent["owned_units"]}
    deadline = config["phase_deadline_epoch"]
    if (parent["state"] != "active" or parent["interval_id"] != config["parent_id"]
            or unit not in registered or type(deadline) not in (int, float)
            or not now < deadline <= parent["approved_until_epoch"]):
        raise ValueError("Missing active registered GPU parent or expired phase")
    bindings = dict(zip(parent["gpu_indices"], parent["gpu_uuids"], strict=True))
    if bindings.get(config["gpu_index"]) != config["gpu_uuid"]:
        raise ValueError("GPU is outside the exact approved parent binding")
    if (parent["cpu_affinity"] != [0, 1, 2, 3]
            or parent["memory_max_bytes"] != 16 * 1024**3):
        raise ValueError("CPU/RAM policy changed; re-admission required")
    return deadline

"""Store completion barrier without a dependency on a particular Torch ABI."""

import hashlib
import json
from pathlib import Path


def checked_store(config):
    root = Path(config["storage_path"]).resolve(strict=True)
    model = config["store_model_path"]
    if not model.startswith("transformers/") or ".." in Path(model).parts:
        raise ValueError("Only an existing HF-layout Store path is admitted")
    path = (root / model).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Store path escaped its approved root")
    index = path / "tensor_index.json"
    if hashlib.sha256(index.read_bytes()).hexdigest() != config["tensor_index_sha256"]:
        raise ValueError("Store tensor index identity changed")
    model_config = json.loads((Path(config["model_path"]) / "config.json").read_text())
    if model_config.get("architectures") not in (["Qwen2ForCausalLM"], ["Qwen3ForCausalLM"]):
        raise ValueError("Initial candidate admits only Qwen2/Qwen3 unquantized TP1")
    if model_config.get("quantization_config"):
        raise ValueError("Quantized Store loading is unqualified")
    return str(root), model



def confirmed_tensors(model_path, storage_path, load, confirm):
    replica, state = load(model_path, {"": 0}, storage_path=storage_path)
    # Existing Store returns None on success and False on RPC failure.
    if confirm(model_path, replica) is False:
        raise RuntimeError("Store rejected GPU transfer completion")
    if not state or any(not getattr(tensor, "is_cuda", False)
                        for tensor in state.values()):
        raise RuntimeError("Store must supply nonempty CUDA tensors; no fallback")
    yield from state.items()

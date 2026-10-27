"""SGLang loader for pre-existing HF-layout ServerlessLLM Store tensors.

Import this module only inside the exact modern SGLang interpreter. The Store
extension must be built for that Python/Torch ABI before launch qualification.
"""

import json
import os

from sglang.srt.model_loader.loader import DefaultModelLoader

from .store_loading import checked_store, confirmed_tensors




class ServerlessStoreLoader(DefaultModelLoader):
    """Use native SGLang model construction/packing, exclusively Store weights."""

    def __init__(self, load_config):
        super().__init__(load_config)
        self.config = json.loads(os.environ["SLLM_SGLANG_CONFIG_JSON"])
        self.storage_path, self.store_path = checked_store(self.config)

    def download_model(self, model_config):
        checked_store(self.config)

    def resolve_model_weights(self, model_config, model):
        raise RuntimeError("Safetensors prefetch/overlap is forbidden for Store loading")

    def _get_all_weights(self, model_config, model):
        if getattr(model, "secondary_weights", ()):
            raise RuntimeError("Secondary checkpoint weights are unqualified")
        import torch
        from sllm_store.client import SllmStoreClient
        from sllm_store.torch import load_dict_non_blocking

        client = SllmStoreClient()
        for name, tensor in confirmed_tensors(
                self.store_path, self.storage_path, load_dict_non_blocking,
                client.confirm_model_loaded):
            if str(tensor.dtype) not in self.config["allowed_tensor_dtypes"]:
                raise RuntimeError("Store dtype not present in the pinned source checkpoint")
            yield name, tensor

    def _prepare_weights(self, *args, **kwargs):
        raise RuntimeError("HF checkpoint fallback is forbidden")

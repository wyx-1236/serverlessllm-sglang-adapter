# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import json
import os
import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

import peft
import torch
import ray
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.generation.streamers import BaseStreamer

from sllm.backends.backend_utils import BackendStatus, SllmBackend
from sllm.logger import init_logger
from sllm_store.transformers import load_lora, load_model, save_lora

logger = init_logger(__name__)


class DeletingException(Exception):
    pass


class InferenceStatus(BaseStreamer):
    def __init__(self, status: BackendStatus):
        super().__init__()
        self.status = status
        self.intermediate = []

    def put(self, value):
        value = value.tolist()
        if not self.intermediate:
            self.intermediate = value
        else:
            # NOTE: This does not support in-flight batching
            # or dynamic batch size
            for i, v in enumerate(value):
                self.intermediate[i].append(v)
        logger.warning(f"Intermediate output: {self.intermediate}")
        if self.status == BackendStatus.DELETING:
            raise DeletingException("Backend is deleting")

    def end(self):
        logger.error("Inference completed")

    def get(self):
        return deepcopy(self.intermediate)

    def delete(self):
        logger.info("Deleting intermediate output")
        self.intermediate = []


class TransformersBackend(SllmBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing TransformersBackend for {model_name} with config: {backend_config}"
        )
        self.model_name = model_name
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.inf_status = InferenceStatus(self.status)
        self.status_lock = threading.Lock()
        self.baseline_lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.past_key_values = None

    def replica_placement(self):
        import ray
        if self.status != BackendStatus.RUNNING or self.model is None:
            raise ValueError('replica model not initialized')
        devices = sorted({str(parameter.device) for parameter in self.model.parameters()})
        if devices != ['cuda:0']:
            raise ValueError('replica parameters must belong to one assigned logical GPU')
        physical_uuid = str(torch.cuda.get_device_properties(0).uuid)
        # Torch 2.9 returns a bare CUuuid; normalize the measured device UUID.
        raw_uuid = physical_uuid[4:] if physical_uuid.startswith('GPU-') else physical_uuid
        try:
            physical_uuid = 'GPU-' + str(uuid.UUID(raw_uuid))
        except (ValueError, AttributeError) as error:
            raise ValueError('physical GPU UUID unavailable') from error
        actor_id = str(ray.get_runtime_context().get_actor_id())
        if len(actor_id) != 32:
            raise ValueError('Ray actor identity unavailable')
        self._replica_identity = {'serving_physical_gpu_uuid': physical_uuid,
                                  'serving_actor_id': actor_id}
        return dict(self._replica_identity, parameter_devices=devices,
                    store_address=os.environ.get('SLLM_STORE_ADDRESS'),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    pid=os.getpid())

    def convert_str_to_json(self, json_str):
        try:
            # Parse the JSON string and return the corresponding Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON string: {e}")
            return None

    def init_backend(self) -> None:
        with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            device_map = self.backend_config.get("device_map", "auto")
            torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
            torch_dtype = getattr(torch, torch_dtype)
            hf_model_class = self.backend_config.get("hf_model_class", None)
            if torch_dtype is None:
                logger.warning(
                    f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
                )
                torch_dtype = torch.float16
            if hf_model_class is None:
                logger.error(
                    f"hf_model_class cannot be None. Please provide a valid model class"
                )
                raise ValueError(
                    "hf_model_class cannot be None. Please provide a valid model class"
                )
            quantization_config = self.backend_config.get(
                "quantization_config", None
            )

            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join("transformers", self.model_name)
            self.model = load_model(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
                hf_model_class=hf_model_class,
                quantization_config=quantization_config,
            )
            tokenizer_path = os.path.join(
                storage_path, "transformers", self.model_name, "tokenizer"
            )
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fall back to load from system's cache
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path
                )
            self.status = BackendStatus.RUNNING

    def _tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    def _encoder_tokenize(self, query: str, max_length: int):
        return self.tokenizer(
            query,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")

    def encode(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        def last_token_pool(
            last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            left_padding = (
                attention_mask[:, -1].sum() == attention_mask.shape[0]
            )
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery: {query}"

        model_name = request_data.get("model", "dummy-model")
        task_instruct = request_data.get("task_instruct", "")
        max_length = request_data.get("max_length", 4096)
        query = request_data.get("input", [])

        if not query:
            return {"error": "Missing query in request data"}

        query = [get_detailed_instruct(task_instruct, q) for q in query]

        batch_dict = self._encoder_tokenize(query, max_length)
        with torch.no_grad():
            output = self.model(**batch_dict, output_hidden_states=True)
        embeddings = last_token_pool(
            output.hidden_states[-1], batch_dict["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_tokens = sum([len(self.tokenizer.tokenize(q)) for q in query])
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embeddings[i].tolist(),
                }
                for i in range(len(embeddings))
            ],
            "model": model_name,
            "usage": {
                "query_tokens": query_tokens,
                "total_tokens": query_tokens,
            },
        }

        return response

    def generate(self, request_data: Optional[Dict[str, Any]]):
        if request_data and "baseline_protocol" in request_data:
            from .baseline_request import generate_baseline
            with self.baseline_lock:
                if not hasattr(self, '_replica_identity'):
                    raise ValueError('replica placement must be verified before requests')
                response = generate_baseline(self, request_data)
                response['baseline'].update(self._replica_identity)
                return response
        try:
            with self.status_lock:
                if self.status != BackendStatus.RUNNING:
                    return {"error": "Model not initialized"}

            assert self.model is not None

            model_name = request_data.get("model", "dummy-model")
            messages = request_data.get("messages", [])
            temperature = request_data.get("temperature", 0.7)
            max_tokens = request_data.get("max_tokens", 10)
            lora_adapter_name = request_data.get("lora_adapter_name", None)

            # Preserve tokenizer defaults unless the request explicitly selects a mode.
            template_kwargs = {}
            if "enable_thinking" in request_data:
                if not isinstance(request_data["enable_thinking"], bool):
                    return {"error": "enable_thinking must be a boolean"}
                template_kwargs["enable_thinking"] = request_data["enable_thinking"]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    **template_kwargs,
                )
            except Exception as e:
                if template_kwargs:
                    return {"error": f"Explicit chat template mode failed: {e}"}
                prompt = "\n".join(
                    f"{message['role'].capitalize()}: {message['content']}"
                    for message in messages
                )

            if not prompt:
                return {"error": "Missing prompt in request data"}

            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "streamer": self.inf_status,
            }

            if lora_adapter_name:
                if (
                    not hasattr(self.model, "peft_config")
                    or lora_adapter_name not in self.model.peft_config
                ):
                    return {"error": f"LoRA adapter {lora_adapter_name} not found"}
                generate_kwargs["adapter_names"] = [lora_adapter_name]

            inputs = self._tokenize(prompt)
            prompt_tokens = inputs.input_ids.shape[1]

            # Generate response
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generate_kwargs,
                    )
            except DeletingException:
                logger.info("Backend is shutting down. Aborting request")
                output_tokens = self.inf_status.get()
                self.inf_status.delete()
                return {
                    "preempted": "True",
                    "current_output": output_tokens,
                    "completed_tokens": len(output_tokens[0]) - prompt_tokens,
                }
            except Exception as e:
                logger.error(f"Failed to generate response: {e}")
                raise e
            else:
                output_text = self.tokenizer.decode(
                    outputs[0][prompt_tokens:], skip_special_tokens=True
                )
                total_tokens = len(outputs[0])
                completion_tokens = total_tokens - prompt_tokens
                # FIXME: consider corner case when max_tokens is reached
                finish_reason = (
                    "stop" if completion_tokens < max_tokens else "length"
                )

                # Generate response compatible with OpenAI's API
                response = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output_text,
                            },
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }

                self.inf_status.delete()

                return response
        finally:
            self.inf_status.delete()

    def load_lora_adapter(self, lora_name: str, lora_path: str):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        if (
            hasattr(self.model, "peft_config")
            and lora_name in self.model.peft_config
        ):
            logger.info(f"LoRA adapter {lora_name} already loaded")
            return

        lora_path = os.path.join("transformers", lora_path)
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
        device_map = self.backend_config.get("device_map", "auto")
        torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        if torch_dtype is None:
            logger.warning(
                f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
            )
            torch_dtype = torch.float16
        self.model = load_lora(
            self.model,
            lora_name,
            lora_path,
            device_map=device_map,
            storage_path=storage_path,
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loaded LoRA adapter {lora_name} from {lora_path}")

    def shutdown(self):
        """Abort all requests and shutdown the backend."""
        with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING
            if self.inf_status:
                self.inf_status.status = BackendStatus.DELETING

        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)

        if self.model is not None:
            del self.model

    def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        self.shutdown()

    def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []

        status = self.inf_status.get()
        logger.info(f"Current tokens: {status}")
        return status

    def resume_kv_cache(self, request_datas):
        logger.info(f"Resuming cache for {request_datas}")
        if not request_datas or len(request_datas[0]) <= 1:
            self.past_key_values = None
            return
        try:
            with torch.no_grad():
                device = self.model.device
                # Leave the final source token for resumed generation. A cache
                # covering the entire input leaves no uncached generation input.
                input_ids = torch.tensor(
                    [tokens[:-1] for tokens in request_datas]
                ).to(device)
                output = self.model.generate(
                    input_ids,
                    past_key_values=self.past_key_values,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    return_legacy_cache=True,
                )
                # The generated token is speculative. Only the source output
                # is authoritative; the cache covers the supplied prefix.
                self.past_key_values = output.past_key_values
        except Exception:
            self.past_key_values = None
            raise

    def resume_generate(
        self, request_data: Optional[Dict[str, Any]], current_output
    ):
        try:
            with self.status_lock:
                if self.status != BackendStatus.RUNNING:
                    return {"error": "Model not initialized"}

            assert self.model is not None

            model_name = request_data.get("model", "dummy-model")
            messages = request_data.get("messages", [])
            temperature = request_data.get("temperature", 0.7)
            max_tokens = request_data.get("max_tokens", 10)

            # Preserve tokenizer defaults unless the request explicitly selects a mode.
            template_kwargs = {}
            if "enable_thinking" in request_data:
                if not isinstance(request_data["enable_thinking"], bool):
                    return {"error": "enable_thinking must be a boolean"}
                template_kwargs["enable_thinking"] = request_data["enable_thinking"]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    **template_kwargs,
                )
            except Exception as e:
                if template_kwargs:
                    return {"error": f"Explicit chat template mode failed: {e}"}
                prompt = "\n".join(
                    f"{message['role'].capitalize()}: {message['content']}"
                    for message in messages
                )

            if not prompt:
                return {"error": "Missing prompt in request data"}

            inputs = self._tokenize(prompt)
            prompt_tokens = inputs.input_ids.shape[1]

            # max_tokens remains the original total completion budget.
            completed_tokens = len(current_output[0]) - prompt_tokens
            remaining_tokens = max(0, max_tokens - completed_tokens)
            try:
                with torch.no_grad():
                    device = self.model.device
                    current_output = torch.tensor(current_output).to(device)
                    if remaining_tokens == 0:
                        outputs = current_output
                    else:
                        outputs = self.model.generate(
                            current_output,
                            past_key_values=self.past_key_values,
                            max_new_tokens=remaining_tokens,
                            temperature=temperature,
                            streamer=self.inf_status,
                        )
            except DeletingException:
                logger.error("Backend is shutting down. Aborting request")
                raise DeletingException("Backend is shutting down")
            except Exception as e:
                logger.error(f"Failed to generate response: {e}")
                raise e
            else:
                output_text = self.tokenizer.decode(
                    outputs[0][prompt_tokens:], skip_special_tokens=True
                )
                total_tokens = len(outputs[0])
                completion_tokens = total_tokens - prompt_tokens
                # FIXME: consider corner case when max_tokens is reached
                finish_reason = (
                    "stop" if completion_tokens < max_tokens else "length"
                )

                # Generate response compatible with OpenAI's API
                response = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output_text,
                            },
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }

                self.inf_status.delete()

                return response
        finally:
            self.past_key_values = None
            self.inf_status.delete()

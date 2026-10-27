"""Ray-owned modern SGLang subprocess; resource policy remains external."""

import asyncio
import json
import os
import signal
import time

from .backend_utils import BackendStatus, SllmBackend
from sllm_sglang.config import NO_PLUGINS, verify_parent
from sllm_sglang.protocol import validate_request


class SglangBackend(SllmBackend):
    def __init__(self, model_name, backend_config=None):
        self.model_name = model_name
        self.config = dict(backend_config or {})
        self.status = BackendStatus.UNINITIALIZED
        self.process = None
        self._log = None
        self._lifecycle = asyncio.Lock()

    async def init_backend(self):
        async with self._lifecycle:
            if self.status is not BackendStatus.UNINITIALIZED:
                raise RuntimeError("Backend initialization is not repeatable")
            deadline = verify_parent(self.config)
            if self.config["model_name"] != self.model_name:
                raise ValueError("Runtime configuration model mismatch")
            if self.process is not None:
                raise RuntimeError("Previous child must be independently reaped")
            from sllm_sglang.worker import check_supervision

            check_supervision(self.config)
            assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if not ({str(self.config["gpu_index"]), self.config["gpu_uuid"]} & set(assigned)):
                raise ValueError("Ray GPU assignment disagrees with approved GPU")
            env = {k: v for k, v in os.environ.items()
                   if k in {"HOME", "USER", "LANG", "PATH"}}
            env.update(self.config["runtime_environment"])
            if self.config["runtime_mode"] != "standalone":
                raise ValueError("Research profile qualifies standalone GPU mode only")
            env.update(CUDA_VISIBLE_DEVICES=self.config["gpu_uuid"],
                       SGLANG_PLUGINS=NO_PLUGINS, HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       SLLM_STORE_ADDRESS=self.config["store_address"],
                       STORAGE_PATH=self.config["storage_path"],
                       SLLM_SGLANG_CONFIG_JSON=json.dumps(self.config),
                       PYTHONPATH=os.pathsep.join(self.config["runtime_pythonpath"]))
            self._log = open(self.config["runtime_log"], "xb", buffering=0)
            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.config["runtime_python"], "-m", "sllm_sglang.worker",
                    env=env, stdout=self._log, stderr=self._log,
                    start_new_session=True)
                startup_end = min(deadline, time.time() + self.config["startup_timeout_seconds"])
                while time.time() < startup_end:
                    verify_parent(self.config)
                    if self.process.returncode is not None:
                        raise RuntimeError("SGLang child exited during startup; inspect runtime log")
                    try:
                        reader, writer = await asyncio.open_unix_connection(self.config["socket_path"])
                    except (FileNotFoundError, ConnectionRefusedError):
                        await asyncio.sleep(0.2)
                        continue
                    try:
                        writer.write(b'{"op":"health"}\n')
                        await writer.drain()
                        ready = json.loads(await asyncio.wait_for(reader.readline(), 2))
                        if ready != {"type": "ready", "model": self.model_name}:
                            raise RuntimeError("Unexpected engine readiness identity")
                    finally:
                        writer.close()
                        await writer.wait_closed()
                    self.status = BackendStatus.RUNNING
                    return
                raise TimeoutError("SGLang startup exhausted its admitted window")
            except BaseException:
                await self._terminate_child()
                raise

    async def generate(self, request_data):
        writer = None
        try:
            if self.status is not BackendStatus.RUNNING:
                raise RuntimeError("Model is not ready")
            verify_parent(self.config)
            data = dict(request_data)
            channel = data.pop("_stream_queue", None)
            validate_request(data)
            if (data["model"] != self.model_name
                    or data["eos_token_id"] != self.config["eos_token_id"]
                    or data["pad_token_id"] != self.config["pad_token_id"]):
                raise ValueError("Request identity/EOS/PAD differs from the frozen model")
            reader, writer = await asyncio.open_unix_connection(
                self.config["socket_path"], limit=128 * 1024)
            writer.write((json.dumps(data) + "\n").encode())
            await writer.drain()
            async def consume():
                while line := await reader.readline():
                    event = json.loads(line)
                    if event.get("request_id") != data["request_id"]:
                        raise ValueError("Stream request identity mismatch")
                    if event["type"] in {"token_chunk", "text_chunk"}:
                        if channel is not None:
                            await asyncio.to_thread(channel.put, event, timeout=3)
                    elif event["type"] == "result":
                        return event["response"]
                    elif event["type"] == "error":
                        raise RuntimeError(event["error"])
                    else:
                        raise ValueError("Unknown stream event")
                raise ValueError("Engine disconnected without a terminal result")

            return await asyncio.wait_for(consume(), timeout=min(
                self.config["request_timeout_seconds"],
                self.config["phase_deadline_epoch"] - time.time()))
        except Exception as error:
            return {"error": type(error).__name__ + ": " + str(error)}
        finally:
            # EOF also notifies the worker to abort when publication fails.
            if writer is not None:
                writer.close()
                await writer.wait_closed()

    async def _terminate_child(self):
        if self.process is not None:
            if self.process.returncode is None:
                pgid = os.getpgid(self.process.pid)
                if pgid != self.process.pid or pgid == os.getpgrp():
                    raise RuntimeError("Refusing to signal an unowned process group")
                os.killpg(pgid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except asyncio.TimeoutError:
                    os.killpg(pgid, signal.SIGKILL)
                    await self.process.wait()
            self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    async def shutdown(self):
        async with self._lifecycle:
            self.status = BackendStatus.DELETING
            await self._terminate_child()

    async def stop(self):
        await self.shutdown()

    async def encode(self, request_data):
        raise NotImplementedError("This cohort admits raw token IDs only")

    async def get_current_tokens(self):
        raise NotImplementedError("In-flight migration is not qualified")

    async def resume_kv_cache(self, request_datas):
        raise NotImplementedError("KV resume is not qualified")

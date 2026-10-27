"""Modern interpreter entry point, guarded before Engine/CUDA construction."""

import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from .config import NO_PLUGINS, engine_arguments, verify_parent
from .runtime import MAX_FRAME_BYTES, serve_request


def check_supervision(config):
    verify_parent(config)
    unit = config["unit"].removesuffix(".service") + ".service"
    group = Path("/proc/self/cgroup").read_text().splitlines()
    paths = [line.split(":", 2)[2] for line in group if line.startswith("0::")]
    if len(paths) != 1 or unit not in Path(paths[0]).parts:
        raise ValueError("Worker is not inside the registered owned cgroup")
    maximum = (Path("/sys/fs/cgroup") / paths[0].lstrip("/") / "memory.max").read_text().strip()
    if maximum == "max" or int(maximum) > 16 * 1024**3:
        raise ValueError("Worker cgroup does not enforce the RAM budget")
    if not set(os.sched_getaffinity(0)).issubset({0, 1, 2, 3}):
        raise ValueError("Worker CPU affinity exceeds the approved budget")
    folder = Path(config["socket_path"]).parent
    if (folder.is_symlink() or folder.stat().st_uid != os.getuid()
            or folder.stat().st_mode & 0o777 != 0o700):
        raise ValueError("Require an owned private stage directory")
    if shutil.disk_usage(folder).free < 12 * 1024**3 + config["maximum_output_bytes"]:
        raise ValueError("Disk protection floor plus phase output budget not available")


async def run(engine, config):
    active = set()
    handlers = set()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    def accept(reader, writer):
        try:
            remaining = verify_parent(config) - time.time()
        except Exception:
            writer.close()
            return
        task = asyncio.create_task(serve_request(engine, config["model_name"],
            reader, writer, active, min(config["request_timeout_seconds"], remaining)))
        handlers.add(task)
        task.add_done_callback(handlers.discard)

    socket = Path(config["socket_path"])
    if socket.exists() or socket.is_symlink():
        raise ValueError("Stage socket already exists; never overwrite a prior worker")
    server = await asyncio.start_unix_server(accept, str(socket), limit=MAX_FRAME_BYTES)
    os.chmod(socket, 0o600)

    async def monitor():
        while not stop.is_set():
            await asyncio.sleep(1)
            try:
                check_supervision(config)
            except Exception:
                stop.set()
                raise

    monitor_task = asyncio.create_task(monitor())
    try:
        await stop.wait()
        if monitor_task.done():
            await monitor_task
    finally:
        server.close()
        await server.wait_closed()
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        for task in tuple(handlers):
            task.cancel()
        if handlers:
            await asyncio.gather(*tuple(handlers), return_exceptions=True)
        socket.unlink(missing_ok=True)


def main():
    config = json.loads(os.environ["SLLM_SGLANG_CONFIG_JSON"])
    check_supervision(config)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Require the qualified modern Python interpreter")
    if os.environ.get("SGLANG_PLUGINS") != NO_PLUGINS:
        raise ValueError("CrossPool/general plugins must be disabled before import")
    for group in ("sglang.srt.plugins", "sglang.srt.platforms"):
        if any(ep.name == NO_PLUGINS for ep in importlib.metadata.entry_points(group=group)):
            raise RuntimeError("Plugin exclusion sentinel collides with installed plugin")
    for package, version in config["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("Runtime dependency identity changed: " + package)
    if not config["runtime_file_sha256"]:
        raise ValueError("Require exact runtime source identities, not version strings only")
    for filename, expected in config["runtime_file_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen runtime source changed: " + filename)
    # Missing/incorrect Store ABI fails before SGLang creates an Engine.
    from sllm_store.torch import load_dict_non_blocking  # noqa: F401
    from .store_loader import ServerlessStoreLoader, checked_store
    checked_store(config)
    from sglang import Engine

    args = engine_arguments(config)
    args["load_format"] = ServerlessStoreLoader
    engine = Engine(**args)
    try:
        engine.loop.run_until_complete(run(engine, config))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

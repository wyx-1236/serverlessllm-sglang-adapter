"""Bounded stream transport with native request-scoped cancellation."""

import asyncio
import json
import time
import uuid

from .protocol import StreamReceipt, validate_request

MAX_FRAME_BYTES = 128 * 1024


async def send(writer, value):
    payload = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Stream frame exceeded its bounded transport budget")
    writer.write(payload)
    await writer.drain()


async def serve_request(engine, model_name, reader, writer, active, timeout):
    """Forward actual Engine chunks; EOF cancels exactly this native request.

    Each connection owns one request. Native IDs are fixed-length random values
    because SGLang's abort protocol may match an ID prefix. Caller IDs remain in
    outward events, but cannot cancel a different caller's prefix-related ID.
    """
    tasks = []
    caller_id = None
    native_id = None
    try:
        line = await asyncio.wait_for(reader.readline(), min(timeout, 10))
        if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
            raise ValueError("Missing or oversized request frame")
        data = json.loads(line)
        if data == {"op": "health"}:
            await send(writer, {"type": "ready", "model": model_name})
            return
        validate_request(data)
        if data["model"] != model_name:
            raise ValueError("Request model does not match this engine")
        caller_id = data["request_id"]
        if caller_id in active:
            caller_id = None
            raise ValueError("Duplicate active request ID")
        active.add(caller_id)
        native_id = uuid.uuid4().hex
        receipt = StreamReceipt(data)

        async def produce():
            stream = await engine.async_generate(
                input_ids=data["input_ids"], rid=native_id, stream=True,
                sampling_params={"temperature": 0.0,
                    "max_new_tokens": data["max_tokens"],
                    "stop_token_ids": data["eos_token_id"],
                    "ignore_eos": False})
            async for original in stream:
                chunk = dict(original)
                meta = dict(chunk.get("meta_info", {}))
                if meta.get("id") != native_id:
                    raise ValueError("Native stream request identity mismatch")
                meta["native_request_id"] = native_id
                meta["id"] = caller_id
                chunk["meta_info"] = meta
                # Copy native cumulative IDs before any asynchronous publication.
                chunk["output_ids"] = list(chunk.get("output_ids", []))
                event = receipt.push(chunk, time.monotonic_ns())
                if event is not None:
                    await send(writer, event)
            await send(writer, {"type": "result", "request_id": caller_id,
                                "response": receipt.finish()})

        generation = asyncio.create_task(produce())
        disconnected = asyncio.create_task(reader.read(1))
        tasks = [generation, disconnected]
        done, _ = await asyncio.wait(tasks, timeout=timeout,
                                    return_when=asyncio.FIRST_COMPLETED)
        if generation in done:
            await generation
        elif disconnected in done:
            raise ConnectionError("Stream consumer disconnected")
        else:
            raise TimeoutError("Native request exceeded its admitted window")
    except (Exception, asyncio.CancelledError) as error:
        if native_id is not None:
            engine.tokenizer_manager.abort_request(native_id)
        if not isinstance(error, asyncio.CancelledError):
            try:
                await send(writer, {"type": "error", "request_id": caller_id,
                                    "error": type(error).__name__ + ": " + str(error)[:512]})
            except (ConnectionError, OSError):
                pass
        else:
            raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if caller_id is not None:
            active.discard(caller_id)
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass

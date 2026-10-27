"""Bounded generation-time SSE bridge for the explicit raw-token protocol."""

import asyncio
from contextlib import suppress
import json
import time
from queue import Empty


def frame(event):
    return ('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode()


async def stream_events(channel, invoke, request_id, timeout=60, cleanup_timeout=10, model=None):
    task = asyncio.ensure_future(invoke(channel))
    deadline = asyncio.get_running_loop().time() + timeout
    created = int(time.time())
    actor = getattr(channel, 'actor', None)
    actor_id = actor._actor_id.hex() if actor is not None else None
    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError('stream generation deadline')
            try:
                event = await channel.get_async(timeout=min(0.1, remaining))
            except Empty:
                if task.done():
                    result = task.result()
                    break
                continue
            event = {**event, 'queue_actor_id': actor_id}
            if model is not None:
                delta = {'content': event.get('text_delta', '')}
                if event.get('sequence') == 0:
                    delta['role'] = 'assistant'
                event.update(id='chatcmpl-' + request_id, object='chat.completion.chunk',
                    created=created, model=model,
                    choices=[{'index': 0, 'delta': delta, 'finish_reason': None}])
            yield frame(event)
        if not isinstance(result, dict) or 'error' in result:
            raise RuntimeError(str(result))
        terminal = {'type': 'result', 'request_id': request_id, 'response': result}
        if model is not None:
            terminal.update(id='chatcmpl-' + request_id, object='chat.completion.chunk',
                created=created, model=model, usage=result.get('usage'),
                choices=[{'index': 0, 'delta': {'content': result['baseline'].get('stream_text_tail', '')},
                          'finish_reason': result['choices'][0]['finish_reason']}])
        yield frame(terminal)
        yield b'data: [DONE]\n\n'
    except Exception as error:
        yield frame({'type': 'error', 'request_id': request_id,
                     'error': type(error).__name__ + ': ' + str(error)})
    finally:
        # Killing only this request's channel makes its next producer put fail.
        channel.shutdown(force=True)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=cleanup_timeout)
        except (Exception, asyncio.CancelledError):
            task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await task


def streaming_response(body, router):
    import anyio
    from fastapi.responses import StreamingResponse
    from ray.util.queue import Queue
    from sllm_sglang.protocol import validate_request

    data = dict(body)
    data['stream'] = False
    validate_request(data)  # Reject caller-supplied internal fields before creating actors.
    request_id = data.get('request_id')
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError('streaming requires a bounded request_id')

    async def events():
        channel = Queue(maxsize=16, actor_options={
            'num_cpus': 0, 'resources': {'control_node': 0.01}})

        async def invoke(queue):
            internal = {**data, '_stream_queue': queue}
            return await router.inference.remote(internal, 'generate')

        iterator = stream_events(channel, invoke, request_id, model=data["model"])
        try:
            async for event in iterator:
                yield event
        finally:
            # Starlette's disconnect cancellation must not skip queue/producer cleanup.
            with anyio.CancelScope(shield=True):
                await iterator.aclose()

    return StreamingResponse(events(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

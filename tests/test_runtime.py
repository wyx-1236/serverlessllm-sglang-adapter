import asyncio
import json
import unittest

from sllm_sglang.runtime import serve_request
from test_protocol import request


class Engine:
    def __init__(self, hold=False):
        self.hold = hold
        self.aborted = []
        self.started = asyncio.Event()
        self.tokenizer_manager = self

    def abort_request(self, rid):
        self.aborted.append(rid)

    async def async_generate(self, **kwargs):
        self.started.set()

        async def chunks():
            yield {"output_ids": [3], "meta_info": {"id": kwargs["rid"]}}
            if self.hold:
                await asyncio.Future()
            yield {"output_ids": [3, 9], "meta_info": {"id": kwargs["rid"],
                "prompt_tokens": 2, "completion_tokens": 2,
                "finish_reason": {"type": "stop"}}}

        return chunks()


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def connect(self, engine):
        self.handlers = []
        self.active = set()

        def accept(reader, writer):
            task = asyncio.create_task(serve_request(
                engine, "qwen", reader, writer, self.active, 5))
            self.handlers.append(task)

        self.server = await asyncio.start_server(accept, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return await asyncio.open_connection("127.0.0.1", port)

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        await asyncio.wait_for(asyncio.gather(*self.handlers), 3)

    async def test_native_chunks_and_terminal_response_reach_client(self):
        engine = Engine()
        reader, writer = await self.connect(engine)
        writer.write((json.dumps(request()) + "\n").encode())
        await writer.drain()
        events = []
        while line := await reader.readline():
            events.append(json.loads(line))
        writer.close()
        await writer.wait_closed()
        self.assertEqual([e["type"] for e in events],
                         ["token_chunk", "token_chunk", "result"])
        self.assertEqual(events[-1]["response"]["baseline"]["output_ids"], [3, 9])
        self.assertEqual(engine.aborted, [])
        self.assertEqual(self.active, set())

    async def test_disconnect_aborts_exact_native_request(self):
        engine = Engine(hold=True)
        reader, writer = await self.connect(engine)
        writer.write((json.dumps(request()) + "\n").encode())
        await writer.drain()
        await engine.started.wait()
        self.assertEqual(json.loads(await reader.readline())["type"], "token_chunk")
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(asyncio.gather(*self.handlers), 2)
        self.assertEqual(len(engine.aborted), 1)
        self.assertEqual(len(engine.aborted[0]), 32)
        self.assertNotEqual(engine.aborted[0], "r-1")
        self.assertEqual(self.active, set())

    async def test_model_mismatch_never_enters_engine(self):
        engine = Engine()
        reader, writer = await self.connect(engine)
        writer.write((json.dumps(request(model="other")) + "\n").encode())
        await writer.drain()
        self.assertEqual(json.loads(await reader.readline())["type"], "error")
        writer.close()
        await writer.wait_closed()
        self.assertFalse(engine.started.is_set())


if __name__ == "__main__":
    unittest.main()

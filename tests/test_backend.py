import unittest

from sllm.backends import SglangBackend


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_unapproved_initialization_cannot_start_a_process(self):
        backend = SglangBackend("qwen", {"parent_path": "missing-parent.json"})
        with self.assertRaises(FileNotFoundError):
            await backend.init_backend()
        self.assertIsNone(backend.process)

    async def test_not_ready_and_unsupported_operations_fail_explicitly(self):
        backend = SglangBackend("qwen", {})
        result = await backend.generate({})
        self.assertIn("error", result)
        with self.assertRaises(NotImplementedError):
            await backend.resume_kv_cache([[1]])
        with self.assertRaises(NotImplementedError):
            await backend.get_current_tokens()
        await backend.shutdown()
        self.assertIsNone(backend.process)


if __name__ == "__main__":
    unittest.main()

import unittest

from sllm_sglang.store_loading import confirmed_tensors


class Tensor:
    is_cuda = True


class StoreLoadingTests(unittest.TestCase):
    def test_store_confirmation_precedes_consumption(self):
        calls = []

        def load(*args, **kwargs):
            calls.append("load")
            return "replica", {"layer.weight": Tensor()}

        def confirm(path, replica):
            calls.append("confirm")
            self.assertEqual((path, replica), ("transformers/qwen", "replica"))

        weights = confirmed_tensors("transformers/qwen", "/store", load, confirm)
        self.assertEqual(list(weights)[0][0], "layer.weight")
        self.assertEqual(calls, ["load", "confirm"])

    def test_failed_confirmation_and_cpu_fallback_are_rejected(self):
        for state, confirmation in (({"w": Tensor()}, False),
                                    ({"w": object()}, None), ({}, None)):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                list(confirmed_tensors("transformers/qwen", "/store",
                     lambda *a, **kw: ("r", state), lambda *a: confirmation))


if __name__ == "__main__":
    unittest.main()

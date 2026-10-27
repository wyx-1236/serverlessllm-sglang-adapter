import json
from pathlib import Path
import tempfile
import unittest

from sllm_sglang.config import engine_arguments, verify_parent


class ConfigTests(unittest.TestCase):
    def test_cache_and_capacity_are_explicit(self):
        args = engine_arguments(dict(model_path="/models/q", prefix_cache=False,
            kv_tokens=8192, max_running_requests=2, attention_backend="triton"))
        self.assertTrue(args["disable_radix_cache"])
        self.assertEqual(args["max_total_tokens"], 8192)
        self.assertEqual(args["tp_size"], 1)
        self.assertEqual(args["startup_weight_load_mode"], "serial")
        self.assertFalse(args["incremental_streaming_output"])
        with self.assertRaises(ValueError):
            engine_arguments(dict(model_path="/m", prefix_cache="false",
                kv_tokens=8192, max_running_requests=2, attention_backend="triton"))

    def test_expired_unregistered_or_wrong_gpu_parent_is_denied(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"parent.json"
            cfg = dict(parent_path=str(path), parent_id="p", unit="u.service",
                       gpu_index=0, gpu_uuid="GPU-zero", phase_deadline_epoch=190)
            good = dict(state="active", interval_id="p", approved_until_epoch=200,
                        gpu_indices=[0, 1], gpu_uuids=["GPU-zero", "GPU-one"],
                        owned_units=["u"], cpu_affinity=[0, 1, 2, 3],
                        memory_max_bytes=16*1024**3)
            path.write_text(json.dumps(good))
            self.assertEqual(verify_parent(cfg, now=100), 190)
            for changes in ({"state": "unlocked"}, {"approved_until_epoch": 90},
                            {"owned_units": []}, {"gpu_uuids": ["GPU-other"]}):
                path.write_text(json.dumps(dict(good, **changes)))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    verify_parent(cfg, now=100)


if __name__ == "__main__":
    unittest.main()

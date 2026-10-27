import unittest

from sllm_sglang.protocol import StreamReceipt, validate_request


def request(**changes):
    data = dict(baseline_protocol="crosspool-preflight-v1", model="qwen",
                request_id="r-1", input_ids=[1, 2], max_tokens=8,
                do_sample=False, eos_token_id=[9], pad_token_id=9, stream=False)
    return dict(data, **changes)


class ProtocolTests(unittest.TestCase):
    def test_raw_ids_and_explicit_sampling(self):
        self.assertEqual(validate_request(request())["input_ids"], [1, 2])
        for changes in ({"input_ids": [True]}, {"max_tokens": 65},
                        {"eos_token_id": []}, {"request_id": ""},
                        {"do_sample": True}, {"messages": []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_request(request(**changes))

    def test_real_chunks_reconcile_usage_and_keep_chunk_timestamps(self):
        receipt = StreamReceipt(request())
        event = receipt.push({"output_ids": [3, 4], "text": "ab", "meta_info": {}}, 100)
        self.assertEqual(event["token_ids"], [3, 4])
        self.assertEqual(event["type"], "token_chunk")
        self.assertEqual(event["text_delta"], "ab")
        self.assertNotIn("per_token_timestamps", event)
        self.assertIsNone(receipt.push({"output_ids": [3, 4], "meta_info": {}}, 110))
        final = receipt.push({"output_ids": [3, 4, 9], "text": "abc", "meta_info": {
            "prompt_tokens": 2, "completion_tokens": 3,
            "finish_reason": {"type": "stop"}}}, 120)
        self.assertEqual(final["text_delta"], "c")
        self.assertEqual(receipt.finish()["usage"]["total_tokens"], 5)

    def test_delayed_text_is_not_dropped_or_counted_as_an_extra_token(self):
        receipt = StreamReceipt(request())
        receipt.push({"output_ids": [3], "text": "", "meta_info": {}}, 100)
        event = receipt.push({"output_ids": [3], "text": "a", "meta_info": {}}, 110)
        self.assertEqual(event["type"], "text_chunk")
        self.assertEqual(event["token_ids"], [])
        self.assertEqual(event["text_delta"], "a")

    def test_rewritten_tokens_missing_terminal_and_bad_usage_fail(self):
        receipt = StreamReceipt(request())
        receipt.push({"output_ids": [3], "meta_info": {}}, 100)
        with self.assertRaises(ValueError):
            receipt.push({"output_ids": [4], "meta_info": {}}, 120)
        with self.assertRaises(ValueError):
            receipt.finish()
        for meta in ({"prompt_tokens": 2, "completion_tokens": 99,
                      "finish_reason": {"type": "length"}},
                     {"prompt_tokens": 2, "completion_tokens": 1,
                      "finish_reason": {"type": "abort"}}):
            bad = StreamReceipt(request())
            with self.assertRaises(ValueError):
                bad.push({"output_ids": [3], "meta_info": meta}, 100)
                bad.finish()


if __name__ == "__main__":
    unittest.main()

import unittest

from shopping_grpo.evaluation.model_client import OpenAIJSONClient


class DeepSeekThinkingControlTests(unittest.TestCase):
    def _payload(self, model, *, thinking=False):
        client = OpenAIJSONClient(
            model=model,
            base_url="https://example.test/v1",
            api_key="test-key",
            thinking=thinking,
        )
        return client._request_payload([{"role": "user", "content": "test"}])

    def test_current_flash_alias_explicitly_disables_thinking(self):
        payload = self._payload("deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_pro_alias_explicitly_disables_thinking(self):
        payload = self._payload("deepseek-v4-pro")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_thinking_mode_removes_sampling_parameters(self):
        payload = self._payload("deepseek-flash", thinking=True)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

    def test_non_deepseek_model_does_not_receive_vendor_extension(self):
        payload = self._payload("example-model")
        self.assertNotIn("thinking", payload)


if __name__ == "__main__":
    unittest.main()

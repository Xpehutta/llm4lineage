import json
import unittest

from Classes.dellm_classes import DELLMGenerator


class _DummyChatAdapter:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def invoke_messages(self, _messages):
        payload = self.payloads[self.calls]
        self.calls += 1
        return payload


class TestDELLMGenerator(unittest.TestCase):
    def test_generate_knowledge_success(self):
        payload = {
            "knowledge": (
                "Total spend is computed as principal amount plus accrued fee. "
                "Payment code 12 means PayPal and joined_at is stored as unix timestamp."
            ),
            "categories": ["arithmetic_reasoning", "domain_terminology", "formatting_synonyms"],
        }
        generator = DELLMGenerator.__new__(DELLMGenerator)
        generator.max_retries = 1
        generator.max_words = 140
        generator.system_prompt = "system"
        generator.chat_adapter = _DummyChatAdapter([json.dumps(payload)])
        generator.chat_model = None

        result = generator.generate_knowledge(
            question="What is total spend by active customers?",
            schema={"tables": [{"name": "payments", "columns": [{"name": "joined_at"}]}]},
        )
        self.assertNotIn("error", result)
        self.assertTrue(result["knowledge"])
        self.assertEqual(generator.chat_adapter.calls, 1)
        self.assertIn("domain_terminology", result["categories"])

    def test_generate_knowledge_trims_long_output(self):
        generator = DELLMGenerator.__new__(DELLMGenerator)
        generator.max_retries = 1
        generator.max_words = 5
        generator.system_prompt = "system"
        generator.chat_adapter = _DummyChatAdapter(
            [
                json.dumps(
                    {
                        "knowledge": "one two three four five six seven eight",
                        "categories": ["arithmetic_reasoning"],
                    }
                )
            ]
        )
        generator.chat_model = None

        result = generator.generate_knowledge(question="Q", schema={})
        self.assertEqual(result["knowledge"], "one two three four five")

    def test_build_augmented_prompt_uses_provided_knowledge(self):
        generator = DELLMGenerator.__new__(DELLMGenerator)
        generator.max_retries = 1
        generator.max_words = 140
        generator.system_prompt = "system"
        generator.chat_adapter = _DummyChatAdapter([])
        generator.chat_model = None

        result = generator.build_augmented_prompt(
            question="List daily revenue by method",
            schema={"tables": [{"name": "payments"}]},
            knowledge="Method code 12 means PayPal.",
        )
        self.assertIn("DELLM Expert Knowledge:", result["final_prompt"])
        self.assertIn("Method code 12 means PayPal.", result["final_prompt"])
        self.assertEqual(result["categories"], [])

    def test_generate_requires_question(self):
        generator = DELLMGenerator.__new__(DELLMGenerator)
        generator.max_retries = 1
        generator.max_words = 140
        generator.system_prompt = "system"
        generator.chat_adapter = _DummyChatAdapter([])
        generator.chat_model = None

        result = generator.generate_knowledge(question="   ", schema={})
        self.assertIn("error", result)
        self.assertEqual(result["knowledge"], "")


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference_engine as ie


class ChatTok:
    chat_template = "{...}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        body = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return body + "<|assistant|>"


class PlainTok:
    chat_template = None


class FormatPromptTests(unittest.TestCase):
    def test_chat_template_wraps_the_prompt_in_a_user_turn(self):
        out = ie.format_prompt(ChatTok(), "What is LoRA?")
        self.assertEqual(out, "<|user|>What is LoRA?<|end|><|assistant|>")

    def test_system_message_comes_first(self):
        out = ie.format_prompt(ChatTok(), "hi", system="Be brief.")
        self.assertTrue(out.startswith("<|system|>Be brief.<|end|><|user|>hi"))

    def test_raw_mode_and_models_without_a_template_pass_through(self):
        self.assertEqual(ie.format_prompt(ChatTok(), "x", chat=False), "x")
        self.assertEqual(ie.format_prompt(PlainTok(), "x"), "x")


if __name__ == "__main__":
    unittest.main()

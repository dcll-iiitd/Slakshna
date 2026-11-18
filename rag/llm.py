"""The agent's LLM interface. See docs/agentic_rag_plan.md section 4.3.

`complete(prompt, max_new_tokens) -> str` is the only method the plan and
reflect steps call, so they never depend on which backend or model is
actually running (D2: switchable LLM).
"""

import json
import urllib.request

# Temperature 0 and a fixed seed, matching the plan's determinism requirement
# (section 4.2) and rag/generator.py's existing OllamaGenerator.
OLLAMA_OPTIONS = {"temperature": 0, "seed": 0}


class OllamaLLM:
    """The Ollama /api/chat API, as rag/generator.py already calls it."""

    name = "ollama"

    def __init__(self, model, url="http://127.0.0.1:11434", timeout=120):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        options = dict(OLLAMA_OPTIONS, num_predict=max_new_tokens)
        body = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.load(response)
        return reply["message"]["content"]


def hf_generation_kwargs():
    """Greedy decoding kwargs for `model.generate()` (plan section 4.2: deterministic).

    A pure function, independent of the lazy torch/transformers import below,
    so the determinism contract is checkable without either installed.
    """
    return {"do_sample": False}


def _seed_everything(torch_module, seed):
    """Fixes the seed before every generation call (plan section 4.2).

    Takes the already-imported torch module as a parameter, rather than
    importing it itself, so this is testable with a fake stand-in.
    """
    torch_module.manual_seed(seed)


class HfLLM:
    """Hugging Face `transformers`, chat template, greedy decoding, GPU if available.

    `torch` and `transformers` are imported lazily here, inside __init__, so
    importing this module - or using the `ollama` backend - never needs them
    installed.
    """

    name = "hf"

    def __init__(self, model, device=None, seed=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)

    def complete(self, prompt, max_new_tokens):
        _seed_everything(self._torch, self.seed)
        messages = [{"role": "user", "content": prompt}]
        # return_dict=True: apply_chat_template returns a BatchEncoding (input_ids +
        # attention_mask), not a bare tensor - generate() needs both, as keywords.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **hf_generation_kwargs())
        reply_ids = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)


def make_llm(backend, model, **options):
    """Build an LLM backend by name. Extra `options` pass through to its constructor."""
    if backend == "ollama":
        return OllamaLLM(model, **options)
    if backend == "hf":
        return HfLLM(model, **options)
    raise ValueError(f"unknown LLM backend '{backend}' (have: ollama, hf)")

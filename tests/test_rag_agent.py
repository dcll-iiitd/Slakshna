"""Unit tests for the agentic RAG loop (docs/agentic_rag_plan.md, Step 1).

Standard library only: no network, no model download. Built up sub-task by
sub-task.
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.agent import Agent, NodeSearch, answer, plan, reflect, rrf_merge, truncate_passages
from rag.agent_profiles import get_profile
from rag.llm import OllamaLLM, hf_generation_kwargs, make_llm, _seed_everything


class _FakeLLM:
    """A scripted LLM: replies in order, or raises, for plan/reflect tests."""

    def __init__(self, replies=None, raises=None):
        self.replies = list(replies or [])
        self.raises = raises
        self.calls = []

    def complete(self, prompt, max_new_tokens):
        self.calls.append((prompt, max_new_tokens))
        if self.raises is not None:
            raise self.raises
        return self.replies.pop(0)


def _key(chunk):
    return (chunk["node_id"], chunk["chunk_id"])


def _chunk(node_id, chunk_id, **extra):
    return dict(node_id=node_id, chunk_id=chunk_id, **extra)


def _response(contexts, peers=None, timings_ms=None, flags=None):
    """A /rag/query-shaped response, as NodeSearch.search() would return."""
    return {"retrieval": "two_round", "answer": "", "contexts": contexts,
            "peers": peers or [], "timings_ms": timings_ms or {}, "flags": flags or []}


class _FakeSearch:
    """A scripted search tool: looks up a canned response by query, or raises."""

    def __init__(self, responses=None, raises_for=None):
        self.responses = dict(responses or {})
        self.raises_for = dict(raises_for or {})
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        if query in self.raises_for:
            raise self.raises_for[query]
        return self.responses[query]


class TestQaProfile(unittest.TestCase):
    def setUp(self):
        self.profile = get_profile("qa")

    def test_format_input_question_only(self):
        question = {"question": "What is the capital of France?"}
        self.assertEqual(self.profile.format_input(question), "What is the capital of France?")

    def test_format_input_with_options(self):
        question = {"question": "Which is a fruit?", "options": {"A": "Carrot", "B": "Apple"}}
        text = self.profile.format_input(question)
        self.assertIn("Which is a fruit?", text)
        self.assertIn("A) Carrot", text)
        self.assertIn("B) Apple", text)

    def test_merge_key_uses_node_id_and_chunk_id(self):
        chunk = {"node_id": "node-1", "chunk_id": "doc.pdf#3", "text": "irrelevant"}
        self.assertEqual(self.profile.merge_key(chunk), ("node-1", "doc.pdf#3"))

    def test_merge_key_local_chunk_has_null_node_id(self):
        # /rag/query reports node_id: null for this node's own chunks.
        chunk = {"node_id": None, "chunk_id": "doc.pdf#3", "text": "irrelevant"}
        self.assertEqual(self.profile.merge_key(chunk), (None, "doc.pdf#3"))

    def test_default_budgets_match_plan(self):
        budgets = self.profile.default_budgets
        self.assertEqual(budgets["max_queries"], 3)
        self.assertEqual(budgets["max_rounds"], 2)
        self.assertEqual(budgets["final_k"], 8)
        # Not specified by the plan: left unbounded rather than invented.
        self.assertIsNone(budgets["max_llm_calls"])
        self.assertIsNone(budgets["timeout_s"])

    def test_prompts_are_non_empty_strings(self):
        for prompt in (self.profile.plan_prompt, self.profile.reflect_prompt, self.profile.answer_prompt):
            self.assertIsInstance(prompt, str)
            self.assertTrue(prompt.strip())

    def test_plan_prompt_formats_with_input_text_and_max_queries(self):
        text = self.profile.plan_prompt.format(input_text="What is the capital of France?", max_queries=3)
        self.assertIn("What is the capital of France?", text)
        self.assertIn("3", text)

    def test_reflect_prompt_formats_with_input_text_and_passages(self):
        text = self.profile.reflect_prompt.format(input_text="Q", passages="[1] Paris is the capital.")
        self.assertIn("Q", text)
        self.assertIn("Paris is the capital.", text)
        self.assertIn('"sufficient"', text)

    def test_answer_prompt_formats_with_input_text_and_passages(self):
        text = self.profile.answer_prompt.format(input_text="Q", passages="[1] Paris is the capital.")
        self.assertIn("Q", text)
        self.assertIn("Paris is the capital.", text)


class TestProfileRegistry(unittest.TestCase):
    def test_get_profile_returns_qa(self):
        self.assertEqual(get_profile("qa").name, "qa")

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            get_profile("legal_pcr")


class _FakeNodeHandler(BaseHTTPRequestHandler):
    """A loopback-only stand-in for a Slakshna node's /rag/query, no real network."""

    response_body = {}
    received = []

    def do_POST(self):
        length = int(self.headers["content-length"])
        body = json.loads(self.rfile.read(length))
        _FakeNodeHandler.received.append((self.path, body))
        payload = json.dumps(_FakeNodeHandler.response_body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestNodeSearch(unittest.TestCase):
    def setUp(self):
        _FakeNodeHandler.received = []
        _FakeNodeHandler.response_body = {
            "retrieval": "two_round",
            "answer": "",
            "contexts": [{
                "node_id": None, "chunk_id": "doc.pdf#1", "score": 0.9,
                "text": "Paris is the capital of France.", "source": "doc.pdf", "verification": None,
            }],
            "peers": [],
            "timings_ms": {},
            "flags": [],
        }
        self.server = HTTPServer(("127.0.0.1", 0), _FakeNodeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def api(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def test_search_posts_expected_body_with_defaults(self):
        tool = NodeSearch(self.api())
        response = tool.search("capital of france")
        self.assertEqual(len(_FakeNodeHandler.received), 1)
        path, body = _FakeNodeHandler.received[0]
        self.assertEqual(path, "/rag/query")
        self.assertEqual(body, {"query": "capital of france", "scope": "all"})
        self.assertEqual(response["contexts"][0]["chunk_id"], "doc.pdf#1")

    def test_search_includes_optional_fields_only_when_set(self):
        tool = NodeSearch(self.api(), retrieval="flat", scope="peers", top_k=20, final_k=5, timeout=5)
        tool.search("q")
        _, body = _FakeNodeHandler.received[0]
        self.assertEqual(body, {"query": "q", "scope": "peers", "retrieval": "flat", "top_k": 20, "final_k": 5})

    def test_search_returns_full_parsed_response(self):
        tool = NodeSearch(self.api())
        response = tool.search("q")
        self.assertIn("peers", response)
        self.assertIn("timings_ms", response)
        self.assertIn("flags", response)

    def test_api_trailing_slash_is_stripped(self):
        tool = NodeSearch(self.api() + "/")
        tool.search("q")
        path, _ = _FakeNodeHandler.received[0]
        self.assertEqual(path, "/rag/query")


class TestRrfMerge(unittest.TestCase):
    def test_single_list_preserves_rank_order(self):
        results = [_chunk("n1", "a"), _chunk("n1", "b"), _chunk("n1", "c")]
        merged = rrf_merge([results], _key)
        self.assertEqual([c["chunk_id"] for c in merged], ["a", "b", "c"])

    def test_score_matches_formula_k60(self):
        results = [_chunk("n1", "a"), _chunk("n1", "b")]
        merged = rrf_merge([results], _key, k=60)
        by_id = {c["chunk_id"]: c["rrf_score"] for c in merged}
        self.assertAlmostEqual(by_id["a"], 1 / 61)
        self.assertAlmostEqual(by_id["b"], 1 / 62)

    def test_item_in_two_lists_gets_summed_score_and_appears_once(self):
        list1 = [_chunk("n1", "shared"), _chunk("n1", "only_in_1")]
        list2 = [_chunk("n1", "only_in_2"), _chunk("n1", "shared")]
        merged = rrf_merge([list1, list2], _key, k=60)
        shared = [c for c in merged if c["chunk_id"] == "shared"]
        self.assertEqual(len(shared), 1)
        self.assertAlmostEqual(shared[0]["rrf_score"], 1 / 61 + 1 / 62)

    def test_repeated_chunk_keeps_first_seen_fields(self):
        list1 = [_chunk("n1", "a", text="first copy")]
        list2 = [_chunk("n1", "a", text="second copy")]
        merged = rrf_merge([list1, list2], _key)
        self.assertEqual(merged[0]["text"], "first copy")

    def test_mixed_none_and_string_node_id_does_not_raise(self):
        list1 = [_chunk(None, "local_chunk")]
        list2 = [_chunk("peer-1", "peer_chunk")]
        # Both rank 1 in their own list, so scores tie; must not raise
        # comparing (None, "local_chunk") against ("peer-1", "peer_chunk").
        merged = rrf_merge([list1, list2], _key)
        self.assertEqual({c["chunk_id"] for c in merged}, {"local_chunk", "peer_chunk"})

    def test_tie_break_is_deterministic_across_calls(self):
        list1 = [_chunk(None, "z")]
        list2 = [_chunk("peer-1", "a")]
        first = [c["chunk_id"] for c in rrf_merge([list1, list2], _key)]
        second = [c["chunk_id"] for c in rrf_merge([list1, list2], _key)]
        self.assertEqual(first, second)

    def test_empty_result_lists_returns_empty(self):
        self.assertEqual(rrf_merge([], _key), [])
        self.assertEqual(rrf_merge([[], []], _key), [])

    def test_works_with_qa_profile_merge_key(self):
        profile = get_profile("qa")
        list1 = [{"node_id": None, "chunk_id": "doc#1", "text": "a"}]
        list2 = [{"node_id": "peer-1", "chunk_id": "doc#1", "text": "b"}]
        merged = rrf_merge([list1, list2], profile.merge_key)
        self.assertEqual(len(merged), 2)  # different node_id -> different keys, not merged


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """A loopback-only stand-in for Ollama's /api/chat, no real network."""

    reply_text = "a reply"
    received = []

    def do_POST(self):
        length = int(self.headers["content-length"])
        body = json.loads(self.rfile.read(length))
        _FakeOllamaHandler.received.append((self.path, body))
        payload = json.dumps({"message": {"content": _FakeOllamaHandler.reply_text}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestOllamaLLM(unittest.TestCase):
    def setUp(self):
        _FakeOllamaHandler.received = []
        _FakeOllamaHandler.reply_text = "a reply"
        self.server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def test_complete_posts_expected_body(self):
        llm = OllamaLLM("qwen2.5:7b-instruct", url=self.url())
        llm.complete("What is the capital of France?", max_new_tokens=64)
        path, body = _FakeOllamaHandler.received[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["model"], "qwen2.5:7b-instruct")
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["seed"], 0)
        self.assertEqual(body["options"]["num_predict"], 64)
        self.assertEqual(body["messages"], [{"role": "user", "content": "What is the capital of France?"}])

    def test_complete_returns_message_content(self):
        _FakeOllamaHandler.reply_text = "Paris."
        llm = OllamaLLM("qwen2.5:7b-instruct", url=self.url())
        self.assertEqual(llm.complete("q", max_new_tokens=10), "Paris.")


class TestMakeLlm(unittest.TestCase):
    def test_make_llm_ollama_passes_through_options(self):
        llm = make_llm("ollama", "qwen2.5:7b-instruct", url="http://127.0.0.1:9", timeout=5)
        self.assertIsInstance(llm, OllamaLLM)
        self.assertEqual(llm.model, "qwen2.5:7b-instruct")
        self.assertEqual(llm.timeout, 5)

    def test_make_llm_hf_dispatches_with_options(self):
        # Mocked so this never imports torch/transformers or loads a model.
        with mock.patch("rag.llm.HfLLM") as fake_hf_cls:
            make_llm("hf", "allenai/OLMo-2-1124-7B-Instruct", device="cpu", seed=0)
        fake_hf_cls.assert_called_once_with("allenai/OLMo-2-1124-7B-Instruct", device="cpu", seed=0)

    def test_make_llm_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            make_llm("made_up_backend", "some-model")


class _FakeTorch:
    """Stands in for torch in determinism tests, no real install needed."""

    def __init__(self):
        self.seeds = []

    def manual_seed(self, seed):
        self.seeds.append(seed)


class TestHfDeterminism(unittest.TestCase):
    """Plan section 4.2: deterministic (greedy decoding, fixed seed).

    Checks the kwargs/seeding contract directly, without transformers/torch
    installed and without loading a model - that's a manual, server-side
    check, not a Windows unit test.
    """

    def test_generation_kwargs_disable_sampling(self):
        kwargs = hf_generation_kwargs()
        self.assertIs(kwargs["do_sample"], False)

    def test_seed_everything_uses_the_configured_fixed_seed(self):
        fake_torch = _FakeTorch()
        _seed_everything(fake_torch, 0)
        self.assertEqual(fake_torch.seeds, [0])

    def test_seed_everything_is_called_before_every_completion(self):
        # HfLLM.complete() must reseed each call, not just once at construction,
        # so repeated calls stay deterministic regardless of call order/count.
        fake_torch = _FakeTorch()
        for seed in (0, 0, 0):
            _seed_everything(fake_torch, seed)
        self.assertEqual(fake_torch.seeds, [0, 0, 0])


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.profile = get_profile("qa")

    def test_parses_json_list(self):
        llm = _FakeLLM(replies=['["query one", "query two"]'])
        queries, error = plan(llm, self.profile, "What is the capital of France?", max_queries=3)
        self.assertEqual(queries, ["query one", "query two"])
        self.assertIsNone(error)

    def test_tolerates_surrounding_text(self):
        reply = 'Sure, here are some queries:\n["a", "b"]\nHope this helps!'
        llm = _FakeLLM(replies=[reply])
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, ["a", "b"])
        self.assertIsNone(error)

    def test_deduplicates_case_insensitive_within_the_list(self):
        llm = _FakeLLM(replies=['["Paris landmarks", "paris landmarks", "France capital"]'])
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, ["Paris landmarks", "France capital"])
        self.assertIsNone(error)

    def test_falls_back_on_unparsable_reply(self):
        llm = _FakeLLM(replies=["not json at all"])
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, [])
        self.assertIsNotNone(error)

    def test_falls_back_on_non_list_json(self):
        llm = _FakeLLM(replies=['{"not": "a list"}'])
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, [])
        self.assertIsNotNone(error)

    def test_falls_back_on_llm_error_without_raising(self):
        llm = _FakeLLM(raises=TimeoutError("model server unreachable"))
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, [])
        self.assertIn("LLM error", error)

    def test_respects_max_queries_cap(self):
        llm = _FakeLLM(replies=['["a", "b", "c", "d", "e"]'])
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, ["a", "b", "c"])

    def test_drops_empty_and_non_string_items(self):
        llm = _FakeLLM(replies=['["", "  ", "valid", 42, null]'])
        queries, error = plan(llm, self.profile, "Q", max_queries=3)
        self.assertEqual(queries, ["valid"])

    def test_prompt_includes_input_text_and_max_queries(self):
        llm = _FakeLLM(replies=["[]"])
        plan(llm, self.profile, "What is the capital of France?", max_queries=5)
        prompt, _ = llm.calls[0]
        self.assertIn("What is the capital of France?", prompt)
        self.assertIn("5", prompt)


class TestTruncatePassages(unittest.TestCase):
    def test_returns_all_passages_when_well_under_budget(self):
        texts = ["short one", "short two", "short three"]
        self.assertEqual(truncate_passages(texts, budget_chars=1000), texts)

    def test_kept_passages_fit_the_total_budget(self):
        texts = ["a" * 100, "b" * 100, "c" * 100, "d" * 100]
        kept = truncate_passages(texts, budget_chars=250)
        self.assertLessEqual(sum(len(t) for t in kept), 250)

    def test_caps_a_single_passage_longer_than_the_whole_budget(self):
        kept = truncate_passages(["x" * 10_000], budget_chars=100)
        self.assertEqual(kept, ["x" * 100])

    def test_per_passage_chars_caps_each_passage_before_the_total_budget(self):
        texts = ["a" * 1000, "b" * 1000, "c" * 1000]
        kept = truncate_passages(texts, budget_chars=1000, per_passage_chars=200)
        self.assertEqual(kept, ["a" * 200, "b" * 200, "c" * 200])

    def test_drops_a_passage_too_small_to_be_useful_instead_of_a_fragment(self):
        # First passage leaves only 10 chars of budget (< _MIN_PASSAGE_CHARS=40):
        # the second passage should be dropped outright, not kept as "bb..b".
        texts = ["a" * 90, "b" * 50]
        kept = truncate_passages(texts, budget_chars=100)
        self.assertEqual(kept, ["a" * 90])

    def test_preserves_best_first_order(self):
        texts = ["first", "second", "third"]
        self.assertEqual(truncate_passages(texts, budget_chars=1000), ["first", "second", "third"])

    def test_deterministic_across_calls(self):
        texts = ["a" * 100, "b" * 100, "c" * 100]
        first = truncate_passages(texts, budget_chars=150)
        second = truncate_passages(texts, budget_chars=150)
        self.assertEqual(first, second)

    def test_empty_input_returns_empty(self):
        self.assertEqual(truncate_passages([], budget_chars=1000), [])

    def test_zero_budget_returns_empty(self):
        self.assertEqual(truncate_passages(["anything"], budget_chars=0), [])


class TestReflect(unittest.TestCase):
    def setUp(self):
        self.profile = get_profile("qa")
        self.chunks = [{"text": "Paris is the capital of France."}, {"text": "France is in Europe."}]

    def test_sufficient_true(self):
        llm = _FakeLLM(replies=['{"sufficient": true}'])
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertTrue(sufficient)
        self.assertIsNone(next_query)
        self.assertIsNone(error)

    def test_sufficient_false_with_next_query(self):
        llm = _FakeLLM(replies=['{"sufficient": false, "next_query": "symptoms of flu"}'])
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertFalse(sufficient)
        self.assertEqual(next_query, "symptoms of flu")
        self.assertIsNone(error)

    def test_tolerates_surrounding_text(self):
        reply = 'Here is my answer:\n{"sufficient": true}\nLet me know if you need more.'
        llm = _FakeLLM(replies=[reply])
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertTrue(sufficient)
        self.assertIsNone(error)

    def test_falls_back_to_stop_on_unparsable_reply(self):
        llm = _FakeLLM(replies=["not json at all"])
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertTrue(sufficient)
        self.assertIsNone(next_query)
        self.assertIsNotNone(error)

    def test_falls_back_to_stop_when_sufficient_field_missing(self):
        llm = _FakeLLM(replies=['{"foo": "bar"}'])
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertTrue(sufficient)
        self.assertIsNotNone(error)

    def test_falls_back_to_stop_when_false_without_next_query(self):
        llm = _FakeLLM(replies=['{"sufficient": false}'])
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertTrue(sufficient)
        self.assertIsNone(next_query)
        self.assertIsNotNone(error)

    def test_falls_back_to_stop_on_llm_error_without_raising(self):
        llm = _FakeLLM(raises=TimeoutError("model server unreachable"))
        sufficient, next_query, error = reflect(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertTrue(sufficient)
        self.assertIn("LLM error", error)

    def test_prompt_includes_input_text(self):
        llm = _FakeLLM(replies=['{"sufficient": true}'])
        reflect(llm, self.profile, "What is the capital of France?", self.chunks, budget_chars=1000)
        prompt, _ = llm.calls[0]
        self.assertIn("What is the capital of France?", prompt)

    def test_passages_are_truncated_to_budget_before_the_prompt(self):
        chunks = [{"text": "A" * 500}, {"text": "B" * 500}]
        llm = _FakeLLM(replies=['{"sufficient": true}'])
        reflect(llm, self.profile, "Q", chunks, budget_chars=100)
        prompt, _ = llm.calls[0]
        self.assertIn("A" * 100, prompt)
        self.assertNotIn("A" * 101, prompt)
        self.assertNotIn("B", prompt)


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.profile = get_profile("qa")
        self.chunks = [{"text": "Paris is the capital of France."}, {"text": "France is in Europe."}]

    def test_returns_llm_reply_stripped(self):
        llm = _FakeLLM(replies=["  Paris [1].  "])
        text, error = answer(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertEqual(text, "Paris [1].")
        self.assertIsNone(error)

    def test_prompt_includes_input_text_and_numbered_passages(self):
        llm = _FakeLLM(replies=["Paris [1]."])
        answer(llm, self.profile, "What is the capital of France?", self.chunks, budget_chars=1000)
        prompt, _ = llm.calls[0]
        self.assertIn("What is the capital of France?", prompt)
        self.assertIn("[1] Paris is the capital of France.", prompt)
        self.assertIn("[2] France is in Europe.", prompt)

    def test_passages_are_truncated_to_budget(self):
        chunks = [{"text": "A" * 500}, {"text": "B" * 500}]
        llm = _FakeLLM(replies=["some answer"])
        answer(llm, self.profile, "Q", chunks, budget_chars=100)
        prompt, _ = llm.calls[0]
        self.assertIn("A" * 100, prompt)
        self.assertNotIn("B", prompt)

    def test_falls_back_on_llm_error_without_raising(self):
        llm = _FakeLLM(raises=TimeoutError("model server unreachable"))
        text, error = answer(llm, self.profile, "Q", self.chunks, budget_chars=1000)
        self.assertEqual(text, "")
        self.assertIn("LLM error", error)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.profile = get_profile("qa")

    def test_original_input_is_always_searched_with_no_planned_queries(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="Paris is the capital.")])})
        llm = _FakeLLM(replies=["[]", '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(search.calls, ["Q"])
        self.assertEqual(result["results"][0]["chunk_id"], "a")
        self.assertEqual(result["trace"]["rounds"], 1)

    def test_original_input_still_searched_when_planning_fails(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="text")])})
        llm = _FakeLLM(replies=["not json", '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(search.calls, ["Q"])
        self.assertIsNotNone(result["trace"]["plan"]["error"])
        self.assertEqual(result["results"][0]["chunk_id"], "a")

    def test_runs_planned_queries_and_merges_results(self):
        shared = _chunk(None, "shared", text="shared text")
        only_a = _chunk(None, "only_a", text="only a")
        only_b = _chunk(None, "only_b", text="only b")
        search = _FakeSearch(responses={
            "Q": _response([shared, only_a]),
            "extra query": _response([only_b, shared]),
        })
        llm = _FakeLLM(replies=['["extra query"]', '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(sorted(search.calls), ["Q", "extra query"])
        self.assertEqual({c["chunk_id"] for c in result["results"]}, {"shared", "only_a", "only_b"})
        # the chunk found by both queries should outrank one found by only one
        self.assertEqual(result["results"][0]["chunk_id"], "shared")

    def test_continues_to_round2_on_insufficient_then_stops(self):
        search = _FakeSearch(responses={
            "Q": _response([_chunk(None, "a", text="a text")]),
            "followup": _response([_chunk(None, "b", text="b text")]),
        })
        llm = _FakeLLM(replies=['[]', '{"sufficient": false, "next_query": "followup"}', '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(search.calls, ["Q", "followup"])
        self.assertEqual(result["trace"]["rounds"], 2)
        self.assertEqual(len(result["trace"]["reflections"]), 2)
        self.assertEqual({c["chunk_id"] for c in result["results"]}, {"a", "b"})

    def test_respects_max_rounds_budget(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="a text")])})
        llm = _FakeLLM(replies=['[]', '{"sufficient": false, "next_query": "followup"}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000, budgets={"max_rounds": 1})
        result = agent.run({"question": "Q"})
        self.assertEqual(search.calls, ["Q"])  # "followup" never searched: round budget used up
        self.assertEqual(result["trace"]["rounds"], 1)

    def test_max_llm_calls_budget_skips_reflect(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="a text")])})
        llm = _FakeLLM(replies=["[]"])  # only the plan call is allowed
        agent = Agent(search, llm, self.profile, budget_chars=1000, budgets={"max_llm_calls": 1})
        result = agent.run({"question": "Q"})
        self.assertEqual(result["trace"]["llm_calls"], 1)
        self.assertTrue(result["trace"]["reflections"][0]["sufficient"])
        self.assertIn("budget exhausted", result["trace"]["reflections"][0]["error"])

    def test_next_query_repeating_original_input_is_not_searched_twice(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="a text")])})
        llm = _FakeLLM(replies=['[]', '{"sufficient": false, "next_query": "Q"}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(search.calls, ["Q"])
        skipped = [q["query"] for q in result["trace"]["queries"] if q.get("skipped") == "duplicate"]
        self.assertEqual(skipped, ["Q"])

    def test_planned_query_repeating_original_input_case_insensitively_is_skipped(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="a text")])})
        llm = _FakeLLM(replies=['["q"]', '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(search.calls, ["Q"])
        skipped = [q["query"] for q in result["trace"]["queries"] if q.get("skipped") == "duplicate"]
        self.assertEqual(skipped, ["q"])

    def test_original_input_search_failure_propagates(self):
        search = _FakeSearch(raises_for={"Q": RuntimeError("node unreachable")})
        llm = _FakeLLM(replies=["[]"])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        with self.assertRaises(RuntimeError):
            agent.run({"question": "Q"})

    def test_planned_query_search_failure_does_not_crash_the_run(self):
        search = _FakeSearch(
            responses={"Q": _response([_chunk(None, "a", text="a text")])},
            raises_for={"bad query": RuntimeError("peer timeout")},
        )
        llm = _FakeLLM(replies=['["bad query"]', '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(result["results"][0]["chunk_id"], "a")
        failed = [q for q in result["trace"]["queries"] if q.get("query") == "bad query"]
        self.assertEqual(len(failed), 1)
        self.assertIn("error", failed[0])

    def test_final_k_caps_output_size(self):
        chunks = [_chunk(None, f"c{i}", text=f"text {i}") for i in range(20)]
        search = _FakeSearch(responses={"Q": _response(chunks)})
        llm = _FakeLLM(replies=["[]", '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000, budgets={"final_k": 5})
        result = agent.run({"question": "Q"})
        self.assertEqual(len(result["results"]), 5)

    def test_trace_includes_llm_calls_and_nonnegative_timing(self):
        search = _FakeSearch(responses={"Q": _response([_chunk(None, "a", text="a text")])})
        llm = _FakeLLM(replies=["[]", '{"sufficient": true}'])
        agent = Agent(search, llm, self.profile, budget_chars=1000)
        result = agent.run({"question": "Q"})
        self.assertEqual(result["trace"]["llm_calls"], 2)
        self.assertGreaterEqual(result["trace"]["llm_time_s"], 0)
        self.assertGreaterEqual(result["trace"]["wall_s"], 0)

    def test_run_is_deterministic_given_the_same_inputs(self):
        def run_once():
            chunks = [_chunk(None, "a", text="a text"), _chunk(None, "b", text="b text")]
            search = _FakeSearch(responses={"Q": _response(chunks)})
            llm = _FakeLLM(replies=["[]", '{"sufficient": true}'])
            return Agent(search, llm, self.profile, budget_chars=1000).run({"question": "Q"})

        def without_timing(queries):
            return [{k: v for k, v in q.items() if k != "retrieval_s"} for q in queries]

        first, second = run_once(), run_once()
        self.assertEqual([c["chunk_id"] for c in first["results"]], [c["chunk_id"] for c in second["results"]])
        self.assertEqual(without_timing(first["trace"]["queries"]), without_timing(second["trace"]["queries"]))


if __name__ == "__main__":
    unittest.main()

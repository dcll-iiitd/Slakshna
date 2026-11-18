"""The agentic RAG loop. See docs/agentic_rag_plan.md section 4.

Built sub-task by sub-task (docs/agentic_rag_rules.md: "one step at a time").
This file now holds every piece from section 4.4's file table: the search
tool (NodeSearch), the RRF merge, the plan and reflect steps, and the Agent
orchestration that ties them together.
"""

import json
import time
import urllib.request


class NodeSearch:
    """Calls POST /rag/query on one Slakshna node's API (QueryRequest, src/rag/mod.rs).

    `retrieval`, `top_k` and `final_k` are left out of the request body when
    None, so the node's own defaults apply. Tests use a fake with the same
    `search(query)` method, so the agent loop never needs a real node.
    """

    def __init__(self, api, retrieval=None, scope="all", top_k=None, final_k=None, timeout=120):
        self.api = api.rstrip("/")
        self.retrieval = retrieval
        self.scope = scope
        self.top_k = top_k
        self.final_k = final_k
        self.timeout = timeout

    def search(self, query):
        """Returns the parsed /rag/query response: contexts, peers, timings_ms, flags, answer."""
        body = {"query": query, "scope": self.scope}
        if self.retrieval is not None:
            body["retrieval"] = self.retrieval
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.final_k is not None:
            body["final_k"] = self.final_k
        request = urllib.request.Request(
            f"{self.api}/rag/query",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def rrf_merge(result_lists, merge_key, k=60):
    """Reciprocal rank fusion across several ranked result lists (plan section 4.1).

    Each list in `result_lists` is one search's `contexts`, best first.
    `merge_key(chunk) -> hashable` groups the same passage across lists (the
    qa profile's default is `(node_id, chunk_id)`). A chunk's fused score is
    the sum, over every list it appears in, of `1 / (k + rank)` (rank is
    1-based). Each key is kept once, using the fields of its first-seen
    occurrence plus an added "rrf_score". Sorted by score descending, with a
    string-of-key tie-break so equal scores still sort the same way every
    time (merge keys may mix None and str, which plain tuple comparison
    cannot order).
    """
    scores = {}
    kept = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = merge_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            kept.setdefault(key, chunk)
    merged = []
    for key, score in scores.items():
        item = dict(kept[key])
        item["rrf_score"] = score
        merged.append((key, item))
    merged.sort(key=lambda pair: (-pair[1]["rrf_score"], str(pair[0])))
    return [item for _, item in merged]


def _extract_json(text, open_char, close_char):
    """Tolerant JSON parsing: the LLM reply may carry extra text around the JSON.

    Tries the whole reply first, then the substring between the first
    `open_char` and the last `close_char` (e.g. "[" / "]" for plan, "{" / "}"
    for reflect). Returns None, never raises, if nothing parses.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find(open_char), text.rfind(close_char)
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _dedup_case_insensitive(queries):
    """Drops a query that repeats (case-insensitively) an earlier one in this same list.

    Order-preserving, first occurrence kept. This is only within-list dedup;
    dropping a query that repeats the original input or an earlier round's
    query is the Agent loop's job (it holds the whole session's history).
    """
    seen = set()
    result = []
    for query in queries:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(query)
    return result


def plan(llm, profile, input_text, max_queries, max_new_tokens=256):
    """Ask the LLM for up to `max_queries` follow-up search queries (plan section 4.1 step 1).

    Returns `(queries, error)`. `queries` is a list of distinct (deduped
    against each other only, case-insensitive), non-empty strings, at most
    `max_queries` long, in the order the LLM wrote them. `error` is `None` on
    success. An LLM error or a reply with no parsable JSON list never raises:
    `queries` is `[]` and `error` names what went wrong, so planning falls
    back to no extra queries (plan section 4.2).
    """
    prompt = profile.plan_prompt.format(input_text=input_text, max_queries=max_queries)
    try:
        reply = llm.complete(prompt, max_new_tokens)
    except Exception as e:  # noqa: BLE001 - any LLM failure must fall back, not crash the run
        return [], f"plan: LLM error: {e}"

    data = _extract_json(reply, "[", "]")
    if data is None:
        return [], "plan: reply had no parsable JSON list"
    if not isinstance(data, list):
        return [], "plan: reply JSON was not a list"

    queries = [item.strip() for item in data if isinstance(item, str) and item.strip()]
    queries = _dedup_case_insensitive(queries)
    return queries[:max_queries], None


# A passage cut shorter than this is dropped rather than kept as a near-useless
# fragment (plan section 4.2: passages are truncated to fit the prompt).
_MIN_PASSAGE_CHARS = 40


def truncate_passages(texts, budget_chars, per_passage_chars=None):
    """Fits a list of passage texts within a total character budget.

    A character budget, not a token budget: real tokenization needs a
    model's own tokenizer (transformers, see rag/llm.py HfLLM), and this
    function has to stay standard-library only. Callers pick `budget_chars`
    as a conservative stand-in for the model's token window (roughly 4
    characters per token for English is a common rule of thumb, but it is
    the caller's call, not this function's).

    Best-first order (as passed in) is preserved. If `per_passage_chars` is
    given, every passage is capped to at most that many characters first, so
    one very long passage cannot crowd out the rest. Passages are then kept
    in order while they fit the remaining budget; a passage that would
    overflow it is cut to fill what's left, unless fewer than
    `_MIN_PASSAGE_CHARS` characters of it would remain, in which case it (and
    everything after it) is dropped instead. Pure and deterministic: the same
    input always gives the same output.
    """
    capped = texts if per_passage_chars is None else [t[:per_passage_chars] for t in texts]
    kept = []
    remaining = budget_chars
    for text in capped:
        if remaining <= 0:
            break
        if len(text) <= remaining:
            kept.append(text)
            remaining -= len(text)
        elif remaining >= _MIN_PASSAGE_CHARS:
            kept.append(text[:remaining])
            remaining = 0
        else:
            break
    return kept


def reflect(llm, profile, input_text, chunks, budget_chars, max_new_tokens=128, per_passage_chars=None):
    """Ask the LLM whether the top passages are enough to answer (plan section 4.1 step 4).

    `chunks` are merged results, best first, each a dict with a "text" key.
    Their texts are truncated to `budget_chars` total (section 4.2) before
    being shown to the LLM - the caller picks the budget, since it depends on
    the model actually running (see truncate_passages).

    Returns `(sufficient, next_query, error)`:
    - the LLM said enough evidence: `(True, None, None)`;
    - the LLM said not enough and gave a follow-up query: `(False, "...", None)`;
    - an LLM error, or a reply with no usable JSON: `(True, None, error)` -
      reflection falls back to stop (section 4.2), never raises. `error`
      names what went wrong, for the trace.
    """
    texts = truncate_passages([chunk["text"] for chunk in chunks], budget_chars, per_passage_chars)
    passages = "\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))
    prompt = profile.reflect_prompt.format(input_text=input_text, passages=passages)
    try:
        reply = llm.complete(prompt, max_new_tokens)
    except Exception as e:  # noqa: BLE001 - any LLM failure must fall back, not crash the run
        return True, None, f"reflect: LLM error: {e}"

    data = _extract_json(reply, "{", "}")
    if not isinstance(data, dict):
        return True, None, "reflect: reply had no parsable JSON object"

    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        return True, None, "reflect: reply JSON missing a boolean 'sufficient'"
    if sufficient:
        return True, None, None

    next_query = data.get("next_query")
    if not isinstance(next_query, str) or not next_query.strip():
        return True, None, "reflect: reply said not sufficient but gave no usable next_query"
    return False, next_query.strip(), None


def answer(llm, profile, input_text, chunks, budget_chars, max_new_tokens=256, per_passage_chars=None):
    """Asks the LLM for a short, cited free-text answer from the kept passages (section 4.4's
    file table: scripts/rag_agent.py prints "a cited answer").

    `chunks` are the final kept results, best first; their texts are
    truncated to `budget_chars` total, same as reflect(). Returns
    `(answer_text, error)`: an LLM error never raises, `answer_text` is ""
    and `error` names what went wrong.

    Checking that citations like "[3]" actually point into the verified pool
    (and counting rejected_citations) is plan section 6.2, task 3d - Step 3,
    not here. This just gets the LLM's raw reply.
    """
    texts = truncate_passages([chunk["text"] for chunk in chunks], budget_chars, per_passage_chars)
    passages = "\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))
    prompt = profile.answer_prompt.format(input_text=input_text, passages=passages)
    try:
        reply = llm.complete(prompt, max_new_tokens)
    except Exception as e:  # noqa: BLE001 - any LLM failure must fall back, not crash the caller
        return "", f"answer: LLM error: {e}"
    return reply.strip(), None


class _TimedLLM:
    """Wraps an LLM backend to count calls and total time for the trace (plan section 4.1.6).

    Transparent to plan()/reflect(): same `complete(prompt, max_new_tokens)`
    method. Counts and times every attempted call, including ones that raise,
    since those still cost LLM time and are still "LLM calls" for the budget.
    """

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0
        self.total_s = 0.0

    def complete(self, prompt, max_new_tokens):
        self.calls += 1
        started = time.monotonic()
        try:
            return self.llm.complete(prompt, max_new_tokens)
        finally:
            self.total_s += time.monotonic() - started


class Agent:
    """Plans, searches, merges and reflects until it has enough evidence or a budget runs out.

    See docs/agentic_rag_plan.md section 4.1. `search_tool` needs a
    `search(query) -> response dict` method (NodeSearch or a fake with the
    same method); `llm` needs `complete(prompt, max_new_tokens) -> str`
    (rag/llm.py or a fake).

    `budget_chars` has no default - how many characters of passage text fit a
    model's prompt depends on the model actually running, and the plan gives
    no general-purpose number (only "OLMo-2-7B has a 4,096-token window"), so
    this is left to the caller rather than invented here.

    `budgets` overrides the profile's defaults (max_queries, max_rounds,
    final_k, max_llm_calls, timeout_s); any key left out keeps the profile's
    default.
    """

    def __init__(self, search_tool, llm, profile, budget_chars, budgets=None, per_passage_chars=None,
                 plan_max_new_tokens=256, reflect_max_new_tokens=128):
        self.search_tool = search_tool
        self.llm = llm
        self.profile = profile
        self.budget_chars = budget_chars
        self.per_passage_chars = per_passage_chars
        self.plan_max_new_tokens = plan_max_new_tokens
        self.reflect_max_new_tokens = reflect_max_new_tokens
        self.budgets = dict(profile.default_budgets)
        if budgets:
            self.budgets.update(budgets)

    def run(self, question):
        """Runs the loop for one input. Returns `{"results": [...], "trace": {...}}`.

        `results` is the top `final_k` merged chunks (plan section 4.1.6).
        Raises only if searching the *original* input fails - every other
        failure (a bad LLM reply, an LLM error, a planned/follow-up search
        failing) is caught and recorded in the trace instead, per section 4.2.
        """
        started = time.monotonic()
        timed_llm = _TimedLLM(self.llm)
        budgets = self.budgets
        input_text = self.profile.format_input(question)
        searched = set()
        result_lists = []
        trace = {"queries": [], "reflections": []}

        def time_left():
            return budgets["timeout_s"] is None or (time.monotonic() - started) < budgets["timeout_s"]

        def llm_budget_left():
            return budgets["max_llm_calls"] is None or timed_llm.calls < budgets["max_llm_calls"]

        def do_search(query, round_num, required=False):
            """Returns this query's contexts, or None if skipped (duplicate) or failed."""
            key = query.strip().lower()
            if key in searched:
                trace["queries"].append({"query": query, "round": round_num, "skipped": "duplicate"})
                return None
            searched.add(key)
            entry = {"query": query, "round": round_num}
            search_started = time.monotonic()
            try:
                response = self.search_tool.search(query)
            except Exception as e:  # noqa: BLE001 - only a *required* search failure should propagate
                if required:
                    raise
                entry["error"] = f"search: {e}"
                trace["queries"].append(entry)
                return None
            entry["retrieval_s"] = round(time.monotonic() - search_started, 3)
            contexts = response.get("contexts", [])
            entry["keys"] = [self.profile.merge_key(c) for c in contexts]
            peers = response.get("peers", [])
            entry["bytes"] = sum((p.get("bytes_sent") or 0) + (p.get("bytes_received") or 0) for p in peers)
            entry["peers_contacted"] = sum(1 for p in peers if (p.get("bytes_sent") or 0) > 0)
            trace["queries"].append(entry)
            return contexts

        # Round 1: plan, then the original input (always, required) plus each planned query.
        if llm_budget_left() and time_left():
            planned, plan_error = plan(timed_llm, self.profile, input_text, budgets["max_queries"],
                                       self.plan_max_new_tokens)
        else:
            planned, plan_error = [], "plan: skipped, budget exhausted before planning"
        trace["plan"] = {"queries": planned, "error": plan_error}

        contexts = do_search(input_text, round_num=1, required=True)
        if contexts is not None:
            result_lists.append(contexts)
        for query in planned:
            contexts = do_search(query, round_num=1, required=False)
            if contexts is not None:
                result_lists.append(contexts)

        merged = rrf_merge(result_lists, self.profile.merge_key)
        round_num = 1
        if llm_budget_left() and time_left():
            sufficient, next_query, reflect_error = reflect(
                timed_llm, self.profile, input_text, merged, self.budget_chars,
                self.reflect_max_new_tokens, self.per_passage_chars,
            )
        else:
            sufficient, next_query, reflect_error = True, None, "reflect: skipped, budget exhausted"
        trace["reflections"].append(
            {"round": round_num, "sufficient": sufficient, "next_query": next_query, "error": reflect_error})

        # Repeat: one follow-up search + merge + reflect per round, up to max_rounds.
        while not sufficient and round_num < budgets["max_rounds"] and time_left():
            round_num += 1
            contexts = do_search(next_query, round_num, required=False)
            if contexts is None:  # a duplicate (plan section 4.2) or a failed search: nothing new to add
                break
            result_lists.append(contexts)
            merged = rrf_merge(result_lists, self.profile.merge_key)

            if llm_budget_left() and time_left():
                sufficient, next_query, reflect_error = reflect(
                    timed_llm, self.profile, input_text, merged, self.budget_chars,
                    self.reflect_max_new_tokens, self.per_passage_chars,
                )
            else:
                sufficient, next_query, reflect_error = True, None, "reflect: skipped, budget exhausted"
            trace["reflections"].append(
                {"round": round_num, "sufficient": sufficient, "next_query": next_query, "error": reflect_error})

        trace["rounds"] = round_num
        trace["llm_calls"] = timed_llm.calls
        trace["llm_time_s"] = round(timed_llm.total_s, 3)
        trace["wall_s"] = round(time.monotonic() - started, 3)
        return {"results": merged[:budgets["final_k"]], "trace": trace}

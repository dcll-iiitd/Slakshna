# Decisions

Format: what was decided, who decided it, the date, and a plain-language why.
This file is new to `dcll-iiitd/Slakshna` (see `docs/agentic_rag_rules.md`) and
starts at section A; it does not reuse the lettering of any other repo's log.

## A. Agentic RAG

### A1. Foundational decisions for the whole plan

Made by the user on 13 Nov 2026, before implementation started. Full detail
is in `docs/agentic_rag_plan.md` section 2 (D1-D5); restated here briefly so
this file is the one place that lists every decision:

- **D1 - Python first.** The agent loop starts as a Python client of the node
  API (Step 1). Moving it into the Rust node is Step 2. Why: lets the agent
  logic be built and tested on the user's Windows machine, without needing
  the Rust node to build there.
- **D2 - LLM default `allenai/OLMo-2-1124-7B-Instruct`,** switchable to
  another model or backend through configuration only, no code change. Why:
  keeps the agent usable with whatever model/hardware is available, and
  matches the model used in earlier runs.
- **D3 - Peer-side answers ("sovereign mode") are out of scope**, built by
  another student. Why: avoids two people changing the same interfaces at
  once; this plan must not block that work either.
- **D4 - New branch `agentic-rag`** in `dcll-iiitd/Slakshna`, fast-forwarded
  from that repo's `RAG` branch. Nothing committed or pushed without the
  user's approval. Why: keeps this work isolated from the verified
  federated-search code it builds on.
- **D5 - Dataset-independent design.** Steps 1-3 must not hard-code any
  dataset; the real evaluation dataset (IL-TUR legal case retrieval) comes
  in Step 4. Development tests use the MedRAG Textbooks/MedQA setup already
  running on the server. Why: the agent loop must still work once the
  dataset changes in Step 4, without rewriting it.

### A2. Step 1 split into small, ordered sub-tasks

Decided by the user and the assistant, 17 Nov 2026. Step 1 (`rag/agent.py`,
`rag/llm.py`, `rag/agent_profiles.py`, the CLI and the M9 script) was broken
into 11 sub-tasks, each built, tested and stopped-on before the next: task
profile, search tool, RRF merge, LLM interface + `ollama`, `hf` backend, plan
step, passage truncation, reflect step, the `Agent` loop, the CLI, the M9
script. Why: matches `docs/agentic_rag_rules.md`'s "one step at a time,"
keeps each piece testable with fakes before the next depends on it, and
gives the user a natural stopping point to review or redirect after each one.

### A3. Passage truncation is a character budget, not a token budget

Decided by the user and the assistant, 17 Nov 2026, as its own sub-task
(`truncate_passages` in `rag/agent.py`), used by both the reflect step and
the answer step. Why: the plan requires passages to fit the model's context
window (section 4.2), but the agent loop must stay standard-library only
(`docs/agentic_rag_rules.md`), so it has no real tokenizer available. A
character budget is a rough, caller-tuned stand-in instead of a guessed
token-to-character ratio baked into the loop.

### A4. Query dedup is split between `plan()` and the `Agent` loop

Decided by the user and the assistant, 17 Nov 2026. `plan()` only drops a
query that case-insensitively repeats another query in the *same* LLM reply.
Dropping a query that repeats the original input, or an earlier round's
query, is the `Agent` loop's job. Why: `plan()` only ever sees one LLM reply
at a time and has no view of the session's history; only the loop orchestrating
every round does, so only it can enforce "repeated queries are not searched
twice" (plan section 4.2) across the whole session.

### A5. `hf` backend determinism is tested without `torch`/`transformers`

Decided by the assistant, confirmed workable by the user, 17 Nov 2026.
Greedy decoding and seeding are pulled out as two small functions -
`hf_generation_kwargs()` (pure, returns `{"do_sample": False}`) and
`_seed_everything(torch_module, seed)` (takes the already-imported `torch` as
a parameter instead of importing it itself) - so both are unit-testable with
a fake stand-in. Why: lets Step 1's "done when" checklist (passes on the
user's Windows machine, standard library only) be met for the determinism
requirement (plan section 4.2) without installing `transformers`/`torch` or
downloading a model, which is a server-side check instead.

### A6. Only the *original input* search is required to succeed

Decided by the assistant while building the `Agent` loop, 17 Nov 2026. If
searching the original input fails, `Agent.run()` raises; if a planned or
follow-up query's search fails, it is caught and recorded in the trace
instead, and the run continues. Why: plan section 4.2 requires "the agent
never sees less evidence than plain RAG" - so the one search plain RAG would
also do must not fail silently - but one flaky supplementary query should
not discard an otherwise-good answer.

### A7. `scripts/m9/run_agentic.py` writes retrieval only, no answer

Decided by the user, 17 Nov 2026. The M9 script never calls the agent's
`answer()` function; `score` in its contexts file is the merged RRF score;
`extra.trace` is exactly the `Agent`'s own trace, without the fuller
per-peer records (`run_slakshna.py`'s `PEER_FIELDS`) that `evaluate.py` uses
for attack-detection and latency-breakdown analyses elsewhere. Why: M9's
evaluation answers every system's retrieved `contexts` with the same
downstream LLM for a fair comparison, so generating an answer here would be
redundant and inconsistent with that methodology; the user confirmed no
planned experiment needs attack-detection or per-peer latency breakdowns for
the agentic system specifically, so that parity gap is deliberately left
open rather than built speculatively.

### A8. `--budget-chars` defaults to 3000, documented as a rough estimate

Decided by the assistant, 17 Nov 2026, in both `scripts/rag_agent.py` and
`scripts/m9/run_agentic.py`. The `Agent` class itself has no default for
`budget_chars` (the caller must always pass it) since the plan gives no
general-purpose number. The CLI scripts need a usable default to run without
every flag set, so they pick 3000 characters, documented in their `--help`
text as a rough, tunable stand-in, not derived from any specific model. Why:
avoids inventing a number inside the reusable loop while still keeping the
command-line tools usable out of the box.

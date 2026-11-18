"""Task profiles: everything the agentic RAG loop needs that depends on the
dataset.

The agent loop itself (planned for rag/agent.py) must stay dataset-independent:
no dataset names, prompts or id formats inside the loop. A profile bundles the
plan prompt, the reflect prompt, the free-text answer prompt, how one input is
turned into search text, the merge key used to pool results across queries,
and default budgets.

Only the `qa` profile is built here (short question, optional multiple-choice
options). A `legal_pcr` profile (long judgment input, merge key = case id) is
planned for Step 4 and is deliberately not built yet.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class Profile:
    name: str
    plan_prompt: str
    reflect_prompt: str
    answer_prompt: str
    format_input: Callable[[dict], str]
    merge_key: Callable[[dict], Tuple]
    default_budgets: Dict[str, Optional[int]]


def _qa_format_input(question):
    """question: {"question": str, "options": {"A": str, ...}}. options is optional."""
    text = question["question"]
    options = question.get("options")
    if options:
        text += "\n" + "\n".join(f"{key}) {value}" for key, value in options.items())
    return text


def _qa_merge_key(chunk):
    """chunk: a /rag/query context dict. node_id is null for this node's own chunks."""
    return (chunk.get("node_id"), chunk["chunk_id"])


QA_PLAN_PROMPT = (
    "You are planning searches to answer a question using a document search tool.\n\n"
    "Question:\n{input_text}\n\n"
    "Write up to {max_queries} short, distinct search queries that would help find passages "
    "answering this question. Each query should cover a different aspect or sub-topic; do not "
    "just repeat the question.\n"
    "Reply with a JSON list of strings only, for example: [\"query one\", \"query two\"]. "
    "If no extra queries would help, reply with an empty list: []."
)

QA_REFLECT_PROMPT = (
    "You are deciding whether enough evidence has been retrieved to answer a question.\n\n"
    "Question:\n{input_text}\n\n"
    "Retrieved passages:\n{passages}\n\n"
    "If the passages contain enough information to answer the question, reply with exactly:\n"
    "{{\"sufficient\": true}}\n"
    "Otherwise, reply with one short follow-up search query that would help, as exactly:\n"
    "{{\"sufficient\": false, \"next_query\": \"...\"}}\n"
    "Reply with JSON only, no other text."
)

QA_ANSWER_PROMPT = (
    "Answer the question using only the numbered passages below. Cite the passages you used "
    "by their number in square brackets, like [2]. If the passages do not contain the answer, "
    "say so.\n\n"
    "Question:\n{input_text}\n\n"
    "Passages:\n{passages}\n\n"
    "Answer:"
)

# Plan §4.1/§4.3 state defaults for max_queries (3), max_rounds (2) and final_k (8).
# max_llm_calls and timeout_s have no stated default in the plan, so they are
# left unbounded (None) rather than inventing a number.
QA_DEFAULT_BUDGETS = {
    "max_queries": 3,
    "max_rounds": 2,
    "final_k": 8,
    "max_llm_calls": None,
    "timeout_s": None,
}

QA = Profile(
    name="qa",
    plan_prompt=QA_PLAN_PROMPT,
    reflect_prompt=QA_REFLECT_PROMPT,
    answer_prompt=QA_ANSWER_PROMPT,
    format_input=_qa_format_input,
    merge_key=_qa_merge_key,
    default_budgets=QA_DEFAULT_BUDGETS,
)

PROFILES = {"qa": QA}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown task profile '{name}' (have: {', '.join(sorted(PROFILES))})")

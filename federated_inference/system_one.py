"""
System One decisions over federated adapters.

A System One model is not a chat model: given program state and a *closed* set of
allowed answers it returns probabilities over those answers in a single
query, with no token-by-token generation (the Jev style, see
https://typesafe.ai/blog/introducing-system-one-models-and-jev). TypeSafe does not
publish Jev's internals, so this is our own construction on ordinary LLM+LoRA
weights, not a reimplementation of Jev:

  1. The options are listed with single-token labels (" A", " B", ...).
  2. ONE forward pass per adapter; the next-token logits at the last position,
     restricted to the label tokens, give a distribution over the options.
  3. Temperature and debiasing: the restricted softmax is divided by a temperature
     (fit it on labelled decisions before reading the numbers as likelihoods), and
     the options are scored under several cyclic re-orderings and averaged, which
     cancels the well-known bias of LLMs towards particular letters/positions.
  4. The adapters' distributions (one per federated node) are mixed with trust
     weights into the final typed decision.
  5. More options than label tokens: two stages, like Jev's "score independently,
     then make an explicit choice". Chunks are scored first, the federation's
     mixed top options of every chunk become the finalists, and a second pass
     chooses among them.

Pure torch; the model is reached through a ``logits_fn(prompt) -> [V]`` callable so
this module is testable without a real model and without PEFT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

MAX_OPTIONS = 255  # Jev's own limit for a single decision
RULES = ("prob_mean", "logit_mean")
_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_PROB_FLOOR = 1e-12
_MIN_SPACED_LABELS = 26  # enough for A-Z; below this fall back to the bare form
# permutations=0 means "auto": every cyclic shift when there are few options (each
# option then visits every slot equally often, so no option is favoured by its letter), and
# this many evenly spread shifts when there are more (partial cancellation).
AUTO_MAX_PERMUTATIONS = 6

LogitsFn = Callable[[str], torch.Tensor]


@dataclass
class Member:
    """One adapter taking part in the decision."""

    name: str
    logits_fn: LogitsFn
    weight: float = 1.0


@dataclass
class MemberResult:
    name: str
    weight: float
    probabilities: list[float]
    # Probability mass the model put on the label tokens BEFORE restricting to
    # them. Low means it was not following the answer format; treat with care.
    format_mass: float


@dataclass
class Decision:
    choice: int
    label: str
    probabilities: list[float]
    confidence: float  # probability of the chosen option
    margin: float  # chosen minus runner-up
    entropy: float  # nats; ln(n) is "no idea"
    agreement: float  # weight share of members whose own top choice matches
    members: list[MemberResult]
    two_stage: bool = False
    finalists: list[int] = field(default_factory=list)
    min_format_mass: float = 1.0

    def to_dict(self) -> dict:
        return {
            "choice": self.choice,
            "label": self.label,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy": self.entropy,
            "agreement": self.agreement,
            "two_stage": self.two_stage,
            "finalists": self.finalists,
            "min_format_mass": self.min_format_mass,
            "members": [m.__dict__ for m in self.members],
        }


# ---------------------------------------------------------------------------
# Labels and prompt
# ---------------------------------------------------------------------------

def label_vocab(tok) -> list[tuple[str, int]]:
    """Distinct single-token labels this tokenizer can express, e.g. [("A", 330), ...].

    All labels use ONE form: with a leading space (what follows "Answer:", preferred)
    or bare (only if the spaced form cannot label at least 26 options). Mixing the two would put some labels
    off-distribution: e.g. Qwen has single tokens for " A".." z" but " 1" is two
    tokens, so a bare "1" label would be systematically under-scored next to the
    letters."""
    def collect(prefix: str) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        seen: set[int] = set()
        for c in _LABEL_CHARS:
            ids = tok.encode(prefix + c, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append((c, ids[0]))
        return out

    spaced, bare = collect(" "), collect("")
    # Prefer the spaced form whenever it can label a reasonable number of options. Bare
    # letters are single tokens too (Qwen has 62 bare vs 52 spaced), so "most labels"
    # would pick the wrong form: after "Answer:" the model's next token is " A".
    return spaced if len(spaced) >= min(_MIN_SPACED_LABELS, len(bare)) else bare


def _clean(option: str) -> str:
    return " ".join(str(option).split())


def validate_options(options: Sequence[str]) -> list[str]:
    cleaned = [_clean(o) for o in options]
    if not 2 <= len(cleaned) <= MAX_OPTIONS:
        raise ValueError(f"need between 2 and {MAX_OPTIONS} options, got {len(cleaned)}")
    if any(not o for o in cleaned):
        raise ValueError("options must be non-empty")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("options must be distinct")
    return cleaned


def build_prompt(state: str, question: str, options: Sequence[str], label_chars: Sequence[str]) -> str:
    lines = [f"{label_chars[i]}. {o}" for i, o in enumerate(options)]
    return (
        f"{state.strip()}\n\nQuestion: {question.strip()}\n"
        "Choose exactly one option.\nOptions:\n" + "\n".join(lines) + "\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Scoring one adapter on one set of options (<= number of labels)
# ---------------------------------------------------------------------------

def _shifts(n: int, permutations: int) -> list[int]:
    """Evenly spread cyclic shifts, so every option visits different positions.
    0 = auto (see AUTO_MAX_PERMUTATIONS); 1 = a single, undebiased ordering."""
    if permutations == 0:
        permutations = AUTO_MAX_PERMUTATIONS
    p = max(1, min(permutations, n))
    return sorted({(i * n) // p for i in range(p)})


def score_options(
    logits_fn: LogitsFn,
    state: str,
    question: str,
    options: Sequence[str],
    labels: Sequence[tuple[str, int]],
    permutations: int = 0,
    calibration_temperature: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """One adapter's distribution over ``options`` and its mean format mass.

    One forward pass per ordering. ``permutations`` 0 = auto, 1 = no debiasing."""
    n = len(options)
    if n > len(labels):
        raise ValueError(f"{n} options but only {len(labels)} single-token labels")
    if calibration_temperature <= 0 or not math.isfinite(calibration_temperature):
        raise ValueError("calibration_temperature must be positive and finite")

    label_chars = [c for c, _ in labels[:n]]
    label_ids = torch.tensor([i for _, i in labels[:n]])
    acc = torch.zeros(n, dtype=torch.float64)
    mass = 0.0
    shifts = _shifts(n, permutations)
    for s in shifts:
        order = [(i + s) % n for i in range(n)]  # position j shows option order[j]
        prompt = build_prompt(state, question, [options[i] for i in order], label_chars)
        logits = logits_fn(prompt).detach().double().cpu()
        full_lse = torch.logsumexp(logits, dim=-1)
        lab = logits[label_ids] / calibration_temperature
        mass += float(torch.exp(torch.logsumexp(logits[label_ids], dim=-1) - full_lse))
        p_pos = torch.softmax(lab, dim=-1)
        for j, opt_idx in enumerate(order):
            acc[opt_idx] += p_pos[j]
    return acc / len(shifts), mass / len(shifts)


# ---------------------------------------------------------------------------
# Federated mixing and the decision
# ---------------------------------------------------------------------------

def normalise_weights(weights: Sequence[float]) -> torch.Tensor:
    w = torch.tensor(list(weights), dtype=torch.float64)
    if w.numel() == 0 or (w < 0).any() or not torch.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    return w / w.sum()


def mix_probs(rows: torch.Tensor, weights: torch.Tensor, rule: str = "prob_mean") -> torch.Tensor:
    """rows [K, n] probabilities (each sums to 1), weights [K] summing to 1 -> [n].

    prob_mean  : sum_k w_k p_k. A bad member's influence is bounded by its weight.
    logit_mean : normalised weighted geometric mean; a confident member can dominate."""
    if rule not in RULES:
        raise ValueError(f"unknown combine rule {rule!r}; expected one of {RULES}")
    if rule == "prob_mean":
        mixed = (weights[:, None] * rows).sum(dim=0)
    else:
        logp = torch.log(rows.clamp_min(_PROB_FLOOR))
        mixed = torch.softmax((weights[:, None] * logp).sum(dim=0), dim=-1)
    return mixed / mixed.sum()


def _score_all(
    members: Sequence[Member],
    state: str,
    question: str,
    options: Sequence[str],
    labels,
    permutations: int,
    calibration_temperature: float,
) -> tuple[torch.Tensor, list[float]]:
    rows, masses = [], []
    for m in members:
        p, mass = score_options(m.logits_fn, state, question, options, labels, permutations, calibration_temperature)
        rows.append(p)
        masses.append(mass)
    return torch.stack(rows), masses


def decide(
    members: Sequence[Member],
    tok,
    state: str,
    question: str,
    options: Sequence[str],
    rule: str = "prob_mean",
    permutations: int = 0,
    calibration_temperature: float = 1.0,
    label_limit: Optional[int] = None,
) -> Decision:
    """The federated System One decision: every member scores the closed option set
    in one pass, the distributions are mixed with the members' weights."""
    if not members:
        raise ValueError("need at least one member")
    opts = validate_options(options)
    labels = label_vocab(tok)
    if label_limit is not None:
        labels = labels[:label_limit]
    if len(labels) < 2:
        raise ValueError("tokenizer offers fewer than two single-token labels")
    w = normalise_weights([m.weight for m in members])
    n = len(opts)

    two_stage = n > len(labels)
    finalists: list[int] = []
    member_probs = torch.zeros(len(members), n, dtype=torch.float64)
    masses = [1.0] * len(members)

    if not two_stage:
        member_probs, masses = _score_all(members, state, question, opts, labels, permutations, calibration_temperature)
        member_probs = member_probs.clone()
    else:
        # Stage 1: score each chunk independently, keep the federation's best of each.
        limit = len(labels)
        n_chunks = math.ceil(n / limit)
        chunks = [list(range(n))[c::n_chunks] for c in range(n_chunks)]
        keep = max(1, limit // n_chunks)
        mass_acc = [0.0] * len(members)
        for chunk in chunks:
            rows, ms = _score_all(members, state, question, [opts[i] for i in chunk], labels, permutations, calibration_temperature)
            mixed = mix_probs(rows, w, rule)
            top = torch.topk(mixed, min(keep, len(chunk))).indices.tolist()
            finalists += [chunk[t] for t in top]
            mass_acc = [a + b / n_chunks for a, b in zip(mass_acc, ms)]
        finalists = sorted(finalists)
        # Stage 2: an explicit choice among the finalists.
        rows, ms = _score_all(members, state, question, [opts[i] for i in finalists], labels, permutations, calibration_temperature)
        for j, idx in enumerate(finalists):
            member_probs[:, idx] = rows[:, j]
        masses = [(a + b) / 2 for a, b in zip(mass_acc, ms)]

    mixed = mix_probs(member_probs, w, rule)
    order = torch.argsort(mixed, descending=True)
    choice = int(order[0])
    runner_up = float(mixed[order[1]]) if n > 1 else 0.0
    ent = float(-(mixed[mixed > 0] * mixed[mixed > 0].log()).sum())
    agreement = float(sum(float(w[k]) for k in range(len(members)) if int(member_probs[k].argmax()) == choice))

    return Decision(
        choice=choice,
        label=opts[choice],
        probabilities=[float(x) for x in mixed],
        confidence=float(mixed[choice]),
        margin=float(mixed[choice]) - runner_up,
        entropy=ent,
        agreement=agreement,
        members=[
            MemberResult(m.name, float(w[k]), [float(x) for x in member_probs[k]], masses[k])
            for k, m in enumerate(members)
        ],
        two_stage=two_stage,
        finalists=finalists,
        min_format_mass=min(masses),
    )

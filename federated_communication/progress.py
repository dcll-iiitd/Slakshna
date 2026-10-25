"""Parsers for the progress lines a Bhaskera training subprocess prints.

Both `bhaskera.launcher.train` (SFT) and `bhaskera.launcher.rl_train` (GRPO) print one line
per step:

    [epoch 0][step 6] loss=2.2678 ... samples=8 tokens=16384

ml_engine.py counts these lines to know whether any training happened this round; a round with
no matching line yields a zero delta. The loss may be negative or in scientific notation (GRPO
losses are), so the number pattern must allow both.
"""
import re

TRAIN_PROGRESS_RE = re.compile(
    r"\[epoch\s+(\d+)\]\[step\s+(\d+)\]\s+loss=(-?[0-9.]+(?:[eE][-+]?\d+)?)"
    r".*?samples=(\d+)\s+tokens=(\d+)"
)
REWARD_RE = re.compile(r"\breward=(-?[0-9.]+)")
# Supervised (loss-bearing) tokens and learning rate of one optimizer step, used to weight the
# per-round training loss by token count.
SUP_TOKENS_RE = re.compile(r"\bsup_tokens=(\d+)")
LR_RE = re.compile(r"\blr=([0-9.]+(?:[eE][-+]?\d+)?)")

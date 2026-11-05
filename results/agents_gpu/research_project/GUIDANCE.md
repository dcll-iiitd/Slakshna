This research program improves the accuracy of a small language model (Qwen2.5-0.5B-Instruct, greedy decoding)
on grade-school math word problems (GSM8K) by changing ONLY the prompt.

The project has one file a candidate may change: prompt.json, with exactly these keys:
  "system": string (<= 4000 chars)               the system message
  "few_shot": list (<= 8) of {"q": str, "a": str}  worked examples shown before the question
  "user_template": string containing {question} exactly once   how the question is presented
  "max_new_tokens": integer 64..512               generation budget
The final line of the model's reply must be 'Answer: <number>' for the answer to be scored, so keep an
instruction that asks for it (in the system message or the user template).

A good agenda item is ONE concrete change to prompt.json, e.g. "add three worked examples that reason step by
step and end with 'Answer: N'", "ask the model to check its arithmetic before answering", "tell it to write one
equation per line". Items must be different from each other and from earlier ones. A candidate is a complete
new prompt.json (all keys).

Rules that matter for accuracy (learned from earlier candidates):
- Worked examples (few_shot "a") must show the reasoning and END with a line 'Answer: <number>'. Examples whose answer
  is just a bare number teach the model to skip reasoning and make it much WORSE.
- Keep "max_new_tokens" at 300 or more; small values cut the reasoning off before the answer.
- Change one thing relative to the current prompt.json you are shown, keep the rest, and keep the final instruction
  that asks for 'Answer: <number>'.

A valid complete prompt.json with two worked examples looks like this (write yours in the same format, between the
=== FILE: prompt.json === markers, as plain JSON):
{"system": "You are a careful math tutor. Think step by step, then finish with a line 'Answer: <number>'.",
 "few_shot": [
  {"q": "Tom has 3 boxes with 4 pencils each and gives away 5 pencils. How many pencils are left?",
   "a": "3 boxes of 4 pencils is 3 * 4 = 12 pencils. After giving away 5: 12 - 5 = 7.\nAnswer: 7"},
  {"q": "A shirt costs $20 and is 25% off. What is the sale price?",
   "a": "25% of 20 is 0.25 * 20 = 5. The sale price is 20 - 5 = 15.\nAnswer: 15"}],
 "user_template": "{question}\nThink step by step and end with 'Answer: <number>'.",
 "max_new_tokens": 400}

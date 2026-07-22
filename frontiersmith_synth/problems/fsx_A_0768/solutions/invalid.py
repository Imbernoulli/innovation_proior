# TIER: invalid
# Declares a dictionary entry that does NOT match the sequence at the position it is
# referenced from (content mismatch). The evaluator's strict content check rejects this,
# so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
seq = inst["seq"]

# a plausible-looking 2-symbol entry, deliberately NOT equal to seq[0:2] in general
bogus = [(seq[0] + 1) % 40, (seq[1] + 1) % 40]

answer = {
    "dictionary": [bogus],
    "segments": [{"type": "ref", "dict_idx": 0}, {"type": "lit", "len": n - 2}],
}
print(json.dumps(answer))

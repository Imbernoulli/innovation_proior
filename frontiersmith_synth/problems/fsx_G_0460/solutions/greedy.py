# TIER: greedy
# One-lever policy: recognise that the sprites are TRANSLATED and manufacture a few
# shifted gallery copies so a shifted test sprite can find a shifted match.  Uses a
# single fixed shift magnitude (mag=3, covering the observed jitter) and does NOT
# touch noise / brightness, so it captures most of the shift gain but leaves the
# noise/brightness gain (and the oracle's larger copy budget) on the table.
import sys, json

json.load(sys.stdin)
policy = {"ops": [{"type": "shift", "mag": 3, "copies": 3}]}
print(json.dumps(policy))

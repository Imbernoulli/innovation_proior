# TIER: invalid
# Emits an out-of-range intervention level (7) for every future day. The
# menu only has levels 0..3, so the evaluator rejects this answer and scores
# it 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
FUT = inst["future_days"]

print(json.dumps({"levels": [7] * FUT}))

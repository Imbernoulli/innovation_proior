# TIER: invalid
# Dumps every well into a single loop (index 0), ignoring both the throughput
# capacity and the temperature band.  Except for degenerate one-well fields this
# overfills loop 0 (and/or blows the temperature band), so the evaluator rejects
# the layout and scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
print(json.dumps({"assign": [0] * N}))

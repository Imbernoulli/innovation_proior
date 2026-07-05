# TIER: invalid
# Emit a mask of the wrong shape (a single 1x1 cell instead of N x N). The
# grader's strict shape check rejects it -> infeasible -> score 0.
import sys, json
json.load(sys.stdin)
print(json.dumps({"phase": [[0.0]]}))

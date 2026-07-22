# TIER: trivial
# Ignore the pools entirely; scan items 0,1,2,... one at a time. No
# adaptivity, no use of the given weights -- the "do nothing clever" recipe.
import sys, json

inst = json.load(sys.stdin)
n = inst["n_items"]


def rec(lo, hi):
    if hi - lo == 1:
        return {"guess": lo}
    return {"query": [lo], "yes": {"guess": lo}, "no": rec(lo + 1, hi)}


tree = rec(0, n)
print(json.dumps({"tree": tree}))

# TIER: trivial
import sys, json
from collections import Counter


def main():
    inst = json.load(sys.stdin)
    N, K, grid = inst["n"], inst["k"], inst["grid"]
    known = [v for row in grid for v in row if v != -1]
    if known:
        cnt = Counter(known)
        mode = max(range(K), key=lambda k: cnt.get(k, 0))
    else:
        mode = 0
    out = [[(v if v != -1 else mode) for v in row] for row in grid]
    print(json.dumps({"grid": out}))


main()

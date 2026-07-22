# TIER: greedy
# The obvious machinist recipe, applied in two independent passes:
#   1) TREE: attach each feature to its SHALLOWEST allowed datum (minimize how
#      far every feature sits from the master datum -> short reference chains).
#   2) SLOTS: "tighten the features that accumulate the most error" -- spend the
#      k precise slots on the DEEPEST features (longest datum chains).
# Both passes ignore criticality weights and how many pairs share an edge, so the
# budget lands on long PRIVATE chains and never on the shared trunk that actually
# caps the weighted worst case.
import sys

def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    n = int(next(it)); k = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    p = [int(next(it)) for _ in range(n)]
    allowed = []
    for i in range(n):
        d = int(next(it)); allowed.append([int(next(it)) for _ in range(d)])

    par = [-1] * n
    depth = [0] * n
    for i in range(1, n):
        # shallowest allowed datum (all allowed indices < i, so depth known)
        best = min(allowed[i], key=lambda d: (depth[d], d))
        par[i] = best
        depth[i] = depth[best] + 1

    # tighten the deepest features first
    order = sorted(range(1, n), key=lambda i: (-depth[i], i))
    precise = order[:k]

    print(" ".join(map(str, par)))
    print(str(len(precise)) + ((" " + " ".join(map(str, precise))) if precise else ""))

main()

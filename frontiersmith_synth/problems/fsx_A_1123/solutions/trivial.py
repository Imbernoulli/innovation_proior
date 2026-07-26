# TIER: trivial
# Topology-blind: pick k anchors uniformly at random using the SAME seeded
# LCG the evaluator uses internally for its weak baseline reference. Never
# looks at the edge list at all.
import sys, json


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _seeded_permutation(seed, n):
    ni = _rng(seed)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = ni(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def main():
    inst = json.load(sys.stdin)
    n, k, seed = inst["n"], inst["k"], inst["seed"]
    anchors = _seeded_permutation(seed, n)[:k]
    print(json.dumps({"anchors": anchors}))


if __name__ == "__main__":
    main()

# TIER: greedy
# The "obvious" recipe: pick ONE truncation depth and apply it uniformly to every
# multiplier, with plain floor truncation (no compensation trick). Scan candidate
# depths from deepest to shallowest and stop at the first depth for which EVERY
# chain's accumulated error (checked exactly against the given sample data) stays
# within its budget. This never looks at per-chain length or at compensation, so a
# single very long accumulation chain in the mix forces the WHOLE instance down to
# a shallow, safe depth -- wasting area on every short chain that could easily have
# tolerated much deeper truncation.
import sys


def trunc0(x, t):
    return x if t == 0 else (x >> t) << t


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    K = int(next(it)); G = int(next(it)); S = int(next(it))
    TMAX = int(next(it)); COMP_EXTRA = int(next(it))
    area = [int(next(it)) for _ in range(TMAX + 1)]

    chain_len = []
    chain_budget = []
    for _ in range(G):
        L = int(next(it)); Bg = int(next(it))
        chain_len.append(L); chain_budget.append(Bg)

    chains = []  # list of list of (a,b) pairs, grouped by chain
    for L in chain_len:
        chain_positions = []
        for _ in range(L):
            pairs = [(int(next(it)), int(next(it))) for _ in range(S)]
            chain_positions.append(pairs)
        chains.append(chain_positions)

    def chain_err_at(chain_positions, t):
        worst = 0
        for s in range(S):
            exact = 0
            approx = 0
            for a, b in (pos[s] for pos in chain_positions):
                exact += a * b
                approx += trunc0(a, t) * trunc0(b, t)
            worst = max(worst, exact - approx)
        return worst

    best_t = 0
    for t in range(TMAX, -1, -1):
        ok = True
        for gi in range(G):
            if chain_err_at(chains[gi], t) > chain_budget[gi]:
                ok = False
                break
        if ok:
            best_t = t
            break

    out = []
    for _ in range(K):
        out.append("%d 0" % best_t)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

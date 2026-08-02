# TIER: strong
# The insight: bias does not have to accumulate. Floor truncation always rounds a
# product DOWN, so its error is one-sided and grows roughly linearly with chain
# length -- a long chain forces very shallow truncation just to stay in budget.
# Round-to-nearest ("compensated") truncation is unbiased: individual errors point
# in both directions and largely cancel across a long accumulation, so the SAME
# truncation depth stays in budget on chains where floor truncation would not.
#
# Per chain (independently, since each chain has its own budget and its own
# length-driven error-growth curve): scan depths deepest-first once with plain
# floor truncation and once with compensated truncation; take whichever feasible
# (depth, mode) pair yields less area for THIS chain -- compensation costs extra
# area per multiplier, so it is only worth paying on chains where it unlocks a
# strictly deeper, and therefore cheaper, truncation than floor truncation could
# reach on its own.
import sys


def trunc0(x, t):
    return x if t == 0 else (x >> t) << t


def trunc1(x, t):
    return x if t == 0 else ((x + (1 << (t - 1))) >> t) << t


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

    chains = []
    for L in chain_len:
        chain_positions = []
        for _ in range(L):
            pairs = [(int(next(it)), int(next(it))) for _ in range(S)]
            chain_positions.append(pairs)
        chains.append(chain_positions)

    def chain_err_at(chain_positions, t, c):
        worst = 0
        for s in range(S):
            exact = 0
            approx = 0
            for a, b in (pos[s] for pos in chain_positions):
                exact += a * b
                if c == 0:
                    approx += trunc0(a, t) * trunc0(b, t)
                else:
                    approx += trunc1(a, t) * trunc1(b, t)
            d = exact - approx
            worst = max(worst, d if c == 0 else abs(d))
        return worst

    out = []
    for gi in range(G):
        cp = chains[gi]
        Bg = chain_budget[gi]

        best_t0 = None
        for t in range(TMAX, -1, -1):
            if chain_err_at(cp, t, 0) <= Bg:
                best_t0 = t
                break
        best_t1 = None
        for t in range(TMAX, -1, -1):
            if chain_err_at(cp, t, 1) <= Bg:
                best_t1 = t
                break

        cost0 = area[best_t0] if best_t0 is not None else None
        cost1 = (area[best_t1] + COMP_EXTRA) if best_t1 is not None else None

        if cost1 is not None and (cost0 is None or cost1 < cost0):
            chosen_t, chosen_c = best_t1, 1
        else:
            chosen_t, chosen_c = best_t0, 0

        line = "%d %d" % (chosen_t, chosen_c)
        out.extend([line] * chain_len[gi])

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

# TIER: strong
# INSIGHT: the metric is cost PER SUCCESSFUL payment, not cost per attempt.
# Because the ladder is a volume-weighted MEAN of independent per-(segment,
# bucket) ratios, the objective is separable: the global optimum is reached
# by optimizing each (segment, bucket) cell on its own, per issuer segment,
# instead of a single fee-sorted cascade shared by everyone.
#
# For a FIXED cascade (subset+order), the probability the cascade eventually
# succeeds is order-invariant (it is 1 - prod of decline probabilities over
# the whole subset). So for a fixed subset the optimal ORDER is the one that
# minimizes expected cost alone -- a classic exchange argument: rail i should
# precede rail j whenever cost_i/p_i <= cost_j/p_j. We additionally search
# WHICH rails to even include (a short, cheap-looking rail with very low
# auth for this segment may not be worth a slot at all -- it just adds a
# wasted attempt plus its retry surcharge before the cascade reaches a rail
# that actually authorizes). Rails/attempt caps are tiny here, so we search
# every distinct-rail cascade up to the attempt cap directly and keep the
# true minimum -- exact per cell, hence globally optimal for this ladder.
import sys
from itertools import permutations


def main():
    it = sys.stdin.read().split()
    p = 0
    R = int(it[p]); S = int(it[p + 1]); B = int(it[p + 2]); K = int(it[p + 3]); FAILPEN_BPS = int(it[p + 4]); p += 5
    amt = [int(it[p + i]) for i in range(B)]; p += B
    fixedFee = [[0] * B for _ in range(R)]
    pctBps = [[0] * B for _ in range(R)]
    retry = [0] * R
    for r in range(R):
        for b in range(B):
            fixedFee[r][b] = int(it[p]); pctBps[r][b] = int(it[p + 1]); p += 2
        retry[r] = int(it[p]); p += 1
    auth = [[0.0] * R for _ in range(S)]
    for s in range(S):
        for r in range(R):
            auth[s][r] = float(it[p]); p += 1
    vol = [[0] * B for _ in range(S)]
    for s in range(S):
        for b in range(B):
            vol[s][b] = int(it[p]); p += 1

    def fee(r, b):
        return fixedFee[r][b] + pctBps[r][b] * amt[b] / 10000.0

    def ratio_of(seq, s, b):
        failpen = FAILPEN_BPS * amt[b] / 10000.0
        cost = 0.0
        reach = 1.0
        for i, r in enumerate(seq, start=1):
            pr = auth[s][r]
            q = 1.0 - pr
            attempt_cost = fee(r, b) + (retry[r] if i > 1 else 0.0)
            cost += reach * attempt_cost
            reach *= q
        cost += reach * failpen
        succ = 1.0 - reach
        return cost / max(succ, 1e-9)

    def best_cascade(s, b):
        # R and K are small on this ladder (R<=6, K<=4), so we can afford to
        # exhaustively search every distinct-rail cascade up to the attempt
        # cap directly -- guaranteeing the true per-cell optimum rather than
        # trusting the exchange-sort heuristic alone.
        best = None
        best_seq = None
        for L in range(1, min(K, R) + 1):
            for seq in permutations(range(R), L):
                r = ratio_of(seq, s, b)
                if best is None or r < best:
                    best, best_seq = r, list(seq)
        return best_seq

    out = []
    for s in range(S):
        for b in range(B):
            seq = best_cascade(s, b)
            out.append(str(len(seq)) + " " + " ".join(str(r) for r in seq))
    print("\n".join(out))


if __name__ == "__main__":
    main()

# TIER: strong
# Insight: the checker never scores the nominal schedule -- it scores the worst of
# K replays, and each published vector can be a genuine multi-job SUBSYSTEM (two
# different sub-project chains' payloads inflated together), not just one job. So
# the quantity that should drive priority is not a job's nominal duration but its
# WORST POSSIBLE duration under the published sweep: its own duration times the
# largest inflation factor of ANY subsystem that contains it (shared with other
# jobs or not), or 1x if it is in none. This per-job reduction is exact for THIS
# objective (a max over independently-replayed scenarios, never a sum): a job's
# relevant risk is its own worst case across every scenario it can appear in,
# regardless of which other jobs a given shock happens to correlate it with --
# what matters combinatorially is that no job's worst case is discovered too
# late, not which other job shares its shock. Propagating THIS pessimistic
# per-job weight through the same tail-length DAG DP as the textbook heuristic
# gives a "robust tail length": how long the worst chain through this job could
# become under some single published shock, whether that shock owns one job or a
# whole cross-chain subsystem. Ranking by robust tail keeps subsystems that could
# become a long, badly-queued chain from being scheduled behind jobs that are
# safe (never inflated) merely because those safe jobs look longer nominally --
# exactly the combinatorial avoidance the checker rewards, and something no
# amount of nominal-critical-path slack padding achieves.
import sys
from fractions import Fraction


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    dur = [0] * (N + 1)
    preds = [[] for _ in range(N + 1)]
    succs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        d = int(next(it)); p = int(next(it))
        plist = [int(next(it)) for _ in range(p)]
        dur[i] = d
        preds[i] = plist
        for pj in plist:
            succs[pj].append(i)

    worst_factor = [Fraction(1, 1)] * (N + 1)
    for _ in range(K):
        num = int(next(it)); den = int(next(it)); s = int(next(it))
        ids = [int(next(it)) for _ in range(s)]
        f = Fraction(num, den)
        for jid in ids:
            if f > worst_factor[jid]:
                worst_factor[jid] = f

    worst_w = [0] * (N + 1)
    for i in range(1, N + 1):
        infl = dur[i] * worst_factor[i]
        worst_w[i] = -(-infl.numerator // infl.denominator)  # ceil, matches checker

    robust_tail = [0] * (N + 1)
    for i in range(N, 0, -1):
        best_succ = max((robust_tail[s] for s in succs[i]), default=0)
        robust_tail[i] = worst_w[i] + best_succ

    order = sorted(range(1, N + 1), key=lambda j: (-robust_tail[j], j))
    print(" ".join(map(str, order)))


main()

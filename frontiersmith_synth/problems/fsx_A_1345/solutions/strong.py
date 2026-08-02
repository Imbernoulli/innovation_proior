# TIER: strong
# Same weight-floor sweep as greedy for the Blocker side (that part really
# is just arithmetic). The insight is on the Claimer side: instead of
# sampling a few self-play trajectories and hoping they generalize, look for
# an explicit STRUCTURAL witness -- a cell that is the shared endpoint of
# >=2 size-2 lines in this level's family (a genuine double threat: take the
# shared cell first, and whichever single reply the opponent plays, at least
# one of the other size-2 lines through it is still fully open). When such a
# cell exists we build the strategy table by EXPLICIT CASE ANALYSIS over
# every one of the opponent's N-1 possible replies -- a branch-complete
# certificate, not a sample -- which is exactly what the checker's
# exhaustive adversarial replay demands.
import sys
from fractions import Fraction


def floor_weight_ok(line):
    return Fraction(1, 2 ** len(line))


def find_double_threat(N, lines):
    partners = {}
    for line in lines:
        if len(line) == 2:
            a, b = line
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)
    best_hub, best_set = None, None
    for hub, ps in partners.items():
        if len(ps) >= 2 and (best_set is None or len(ps) > len(best_set)):
            best_hub, best_set = hub, ps
    return best_hub, best_set


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    pool = []
    for _ in range(P):
        sz = int(next(it))
        cells = [int(next(it)) for _ in range(sz)]
        pool.append(cells)
    c = [int(next(it)) for _ in range(K)]

    out = [str(K)]
    for j in range(1, K + 1):
        lines = pool[:c[j - 1]]
        total = sum(floor_weight_ok(l) for l in lines)
        if total < Fraction(1, 2):
            weights = [f"1/{2 ** len(l)}" for l in lines]
            out.append(f"{j} B " + " ".join(weights))
            continue

        hub, ps = find_double_threat(N, lines)
        if hub is not None:
            ps_sorted = sorted(ps)
            rows = [f"0 {hub}"]
            for r in range(1, N + 1):
                if r == hub:
                    continue
                # respond with any live partner different from r
                reply = ps_sorted[0] if ps_sorted[0] != r else ps_sorted[1]
                rows.append(f"2 {hub} {r} {reply}")
            out.append(f"{j} M {len(rows)}")
            out.extend(rows)
        else:
            out.append(f"{j} U")
    print("\n".join(out))


if __name__ == "__main__":
    main()

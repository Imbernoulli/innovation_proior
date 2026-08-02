# TIER: strong
# Insight: with a per-event accumulation limit, the loss curve for a storm is
# piecewise-linear CONVEX in the exposure already written into its footprint (a mild
# rate below the limit, a steep penalty rate above it). That means a candidate's
# marginal value is NOT a fixed number -- it depends on how much of each storm's
# footprint the book you have ALREADY written has used up. The same policy is worth a
# lot when written into an empty footprint and worth little (or negative) once that
# footprint is near its limit. So instead of ranking once by isolated margin, we
# repeatedly recompute each remaining candidate's marginal net value against the
# CURRENT book and always take the best one -- an exchange-argument / matroid-style
# lazy greedy over a submodular-ish (concave return / convex-cost) objective, letting
# the solver both take calculated controlled overage when it still pays and steer
# capacity toward storms that are not yet crowded.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); C = int(next(it)); K = int(next(it)); OVER_MULT = int(next(it))
    storms = []
    for _ in range(K):
        cx = int(next(it)); cy = int(next(it)); R = int(next(it)); L = int(next(it)); sev = int(next(it))
        storms.append((cx, cy, R, L, sev))
    policies = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it)); e = int(next(it)); p = int(next(it)); tech = int(next(it))
        policies.append((x, y, e, p, tech))

    footprint = [[] for _ in range(N)]
    for i, (x, y, e, p, tech) in enumerate(policies):
        for s, (cx, cy, R, L, sev) in enumerate(storms):
            if (x - cx) ** 2 + (y - cy) ** 2 <= R * R:
                footprint[i].append(s)

    margin = [policies[i][3] - policies[i][4] for i in range(N)]

    def marginal_loss(i, X):
        e = policies[i][2]
        tot = 0.0
        for s in footprint[i]:
            L = storms[s][3]
            sev = storms[s][4]
            x0 = X[s]
            x1 = x0 + e
            if x1 <= L:
                add = e
            elif x0 >= L:
                add = OVER_MULT * e
            else:
                add = (L - x0) + OVER_MULT * (x1 - L)
            tot += sev * add / 1000.0
        return tot

    X = [0] * K
    book = []
    remaining = set(range(N))
    while len(book) < C and remaining:
        best_i = None
        best_val = 0.0
        for i in sorted(remaining):
            val = margin[i] - marginal_loss(i, X)
            if best_i is None or val > best_val + 1e-12:
                best_i = i
                best_val = val
        if best_i is None or best_val <= 1e-9:
            break
        book.append(best_i)
        remaining.discard(best_i)
        e = policies[best_i][2]
        for s in footprint[best_i]:
            X[s] += e

    print(len(book))
    print(*book)


if __name__ == "__main__":
    main()

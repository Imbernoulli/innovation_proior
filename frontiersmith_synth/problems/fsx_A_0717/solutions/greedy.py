# TIER: greedy
# The obvious textbook approach: classic pooled maximum-coverage greedy. At
# each step, pick the untested x that kills the most NOT-YET-KILLED mutants,
# counting all families together (a point that lies inside three big
# whole-segment mutants at once looks "great" to this objective). It never
# looks at per-family balance, so it happily spends the whole budget racing
# through the families that are NUMEROUS/EASY (const_shift, coef_perturb,
# sign_flip -- each mutant spans an entire segment) and starves the families
# whose mutants are hard to find (interior_anomaly, sparse_anomaly), which can
# leave the minimax objective at, or near, zero even though "most" mutants
# got killed.
import sys


def main():
    toks = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = int(toks[pos[0]])
        pos[0] += 1
        return v

    D = nxt()
    T = nxt()
    K = nxt()
    families = []
    for _ in range(K):
        n_f = nxt()
        fam = [(nxt(), nxt()) for _ in range(n_f)]
        families.append(fam)

    mutants = []  # [fam_idx, lo, hi, killed]
    for f, fam in enumerate(families):
        for lo, hi in fam:
            mutants.append([f, lo, hi, False])

    cand = sorted(set(m[1] for m in mutants))

    xs = []
    for _ in range(T):
        best_x, best_gain = None, -1
        for x in cand:
            gain = sum(1 for m in mutants if not m[3] and m[1] <= x < m[2])
            if gain > best_gain:
                best_gain, best_x = gain, x
        if best_gain <= 0:
            break
        xs.append(best_x)
        for m in mutants:
            if not m[3] and m[1] <= best_x < m[2]:
                m[3] = True

    print(len(xs))
    print(" ".join(map(str, xs)))


if __name__ == "__main__":
    main()

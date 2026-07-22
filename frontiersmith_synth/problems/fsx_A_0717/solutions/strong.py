# TIER: strong
# The reformulation: the objective is a MINIMUM over families, so the only
# marginal test point worth spending is one that helps the CURRENTLY WORST
# family -- a max-min / water-filling exchange argument, not "kill the most
# mutants overall". At every step we identify the family with the smallest
# kill-fraction so far, and among candidate points drawn from ITS uncovered
# mutants we pick the one with the largest full side-effect (it may also,
# for free, kill mutants in OTHER families whose disagreement region happens
# to overlap that x -- e.g. any x inside a segment automatically kills that
# segment's const_shift/coef_perturb/sign_flip mutants too). This directly
# targets each family's SMALLEST discrimination region instead of chasing
# whichever family currently has the most (numerous) mutants.
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

    killed = [set() for _ in range(K)]

    def frac(f):
        n_f = len(families[f])
        return len(killed[f]) / n_f if n_f else 1.0

    xs = []
    for _ in range(T):
        order = sorted(range(K), key=frac)
        chosen = None
        for f in order:
            fam = families[f]
            uncovered = [i for i in range(len(fam)) if i not in killed[f]]
            if not uncovered:
                continue
            best_x, best_gain = None, -1
            for i in uncovered:
                lo, hi = fam[i]
                x = (lo + hi - 1) // 2
                new_gain = 0
                for g in range(K):
                    for j, (lo2, hi2) in enumerate(families[g]):
                        if j not in killed[g] and lo2 <= x < hi2:
                            new_gain += 1
                if new_gain > best_gain or (new_gain == best_gain and (best_x is None or x < best_x)):
                    best_gain, best_x = new_gain, x
            chosen = best_x
            break
        if chosen is None:
            break
        xs.append(chosen)
        for g in range(K):
            for j, (lo2, hi2) in enumerate(families[g]):
                if lo2 <= chosen < hi2:
                    killed[g].add(j)

    print(len(xs))
    print(" ".join(map(str, xs)))


if __name__ == "__main__":
    main()

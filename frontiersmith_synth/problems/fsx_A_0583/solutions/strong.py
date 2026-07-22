# TIER: strong
# The insight: a mixture of a few SHORT laws masquerades as one LONG messy law on
# any finite window.  Restricting the ledger to a residue class  n == r (mod m)
# exposes a clean order-2 system  a(n) = A[r]*a(n-1) + B[r]*a(n-2) + C[r].  So:
#   1. For each candidate modulus m, split the recurrence equations by n mod m.
#   2. In each class recover (A,B,C) ROBUSTLY: solve exact integer 3x3 systems from
#      consecutive equation triples to gather candidates, then pick the candidate
#      with the largest EXACT consensus over the class.  Consensus counting is
#      immune to the corrupted lines (an outlier just fails the check), which is
#      what defeats plain least squares.
#   3. When several laws fit the finite window equally (a short window is
#      genuinely ambiguous), break the tie by PARSIMONY -- the smallest-coefficient
#      law -- the Occam prior that a mixture of SHORT laws is the right reading.
#   4. Pick the smallest m whose every class reaches near-total consensus, then
#      forward-simulate from the clean seed lines a(1),a(2) and read off held-out.
import sys
from fractions import Fraction


def solve3_int(rows, rhs):
    F = [[Fraction(x) for x in rows[i]] + [Fraction(rhs[i])] for i in range(3)]
    for c in range(3):
        piv = None
        for r in range(c, 3):
            if F[r][c] != 0:
                piv = r
                break
        if piv is None:
            return None
        F[c], F[piv] = F[piv], F[c]
        for r in range(3):
            if r != c and F[r][c] != 0:
                f = F[r][c] / F[c][c]
                for k in range(c, 4):
                    F[r][k] -= f * F[c][k]
    sol = [F[i][3] / F[i][i] for i in range(3)]
    if any(s.denominator != 1 for s in sol):
        return None
    return tuple(int(s) for s in sol)


def recover_class(eqs):
    """eqs = list of (a_{n-1}, a_{n-2}, 1, a_n).  Return (best (A,B,C), consensus).
    Max consensus, ties broken by minimal |A|+|B|+|C| (parsimony)."""
    cand = set()
    for i in range(len(eqs) - 2):
        tri = eqs[i:i + 3]
        sol = solve3_int([[e[0], e[1], e[2]] for e in tri], [e[3] for e in tri])
        if sol is not None:
            cand.add(sol)
    if not cand:
        return None, 0

    def consensus(sol):
        A, B, C = sol
        return sum(1 for e in eqs if A * e[0] + B * e[1] + C == e[3])

    best = max(cand, key=lambda s: (consensus(s), -(abs(s[0]) + abs(s[1]) + abs(s[2]))))
    return best, consensus(best)


def main():
    data = sys.stdin.read().split()
    N, K = int(data[0]), int(data[1])
    a = [0] + [int(x) for x in data[3:3 + N]]        # 1-indexed observed

    best_m = None
    best_laws = None
    best_score = -1.0
    for m in (2, 3, 4):
        laws = {}
        total_eq = 0
        total_cons = 0
        good = True
        for r in range(m):
            eqs = [(a[n - 1], a[n - 2], 1, a[n]) for n in range(3, N + 1) if n % m == r]
            if len(eqs) < 5:
                good = False
                break
            coef, cons = recover_class(eqs)
            if coef is None:
                good = False
                break
            laws[r] = coef
            total_eq += len(eqs)
            total_cons += cons
        if not good:
            continue
        frac = total_cons / total_eq
        if frac >= 0.90 and best_m is None:      # smallest m that explains all
            best_m, best_laws, best_score = m, laws, frac
            break
        if frac > best_score:
            best_m, best_laws, best_score = m, laws, frac

    if best_laws is None:
        sys.stdout.write("\n".join(str(a[N]) for _ in range(K)) + "\n")
        return

    m = best_m
    laws = best_laws
    latent = [0] * (N + K + 1)
    latent[1], latent[2] = a[1], a[2]
    for n in range(3, N + K + 1):
        A, B, C = laws[n % m]
        latent[n] = A * latent[n - 1] + B * latent[n - 2] + C

    preds = [latent[N + 1 + i] for i in range(K)]
    sys.stdout.write("\n".join(str(x) for x in preds) + "\n")


if __name__ == "__main__":
    main()

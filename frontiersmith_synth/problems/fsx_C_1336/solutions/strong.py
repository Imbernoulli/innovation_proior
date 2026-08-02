# TIER: strong
# The insight: don't proxy-maximize hydrogen-bond strength -- directly
# evaluate the actual trade-off. Brute-force every regulatory-approved
# (coformer, ratio) pair (the search space is small -- "scale": small), keep
# only those that clear the stability floor, and pick the one that truly
# maximizes the solubility-improvement formula (polarity bonus minus
# lattice-energy-excess penalty). This is the reformulation that finds the
# interior optimum: stable enough to form, weak enough to dissolve.
import sys


def h_score(W, Ad, Aa, fd, fa, r, K):
    tot = 0
    for t in range(K):
        tot += W[t] * (min(Ad[t], r * fa[t]) + min(Aa[t], r * fd[t]))
    return tot


def main():
    tok = sys.stdin.read().split()
    p = 0

    def nxt(cnt):
        nonlocal p
        v = tok[p:p + cnt]
        p += cnt
        return v

    K = int(tok[p]); p += 1
    donor_strength = list(map(int, nxt(K)))
    acceptor_strength = list(map(int, nxt(K)))
    Ad = list(map(int, nxt(K)))
    Aa = list(map(int, nxt(K)))
    P_BONUS, DECAY, L_min = map(int, nxt(3))
    M = int(tok[p]); p += 1
    W = [donor_strength[t] * acceptor_strength[t] for t in range(K)]

    formers = []
    for _ in range(M):
        fd = list(map(int, nxt(K)))
        fa = list(map(int, nxt(K)))
        lc = int(tok[p]); p += 1
        pol = int(tok[p]); p += 1
        R = int(tok[p]); p += 1
        ratios = list(map(int, nxt(R)))
        formers.append((fd, fa, lc, pol, ratios))

    best_val = None
    best_pick = (0, 1)
    for idx, (fd, fa, lc, pol, ratios) in enumerate(formers):
        for r in ratios:
            h = h_score(W, Ad, Aa, fd, fa, r, K)
            L = h + lc * r
            if L < L_min:
                continue
            dSol = P_BONUS * pol - DECAY * (L - L_min)
            if dSol <= 0:
                continue
            if best_val is None or dSol > best_val:
                best_val = dSol
                best_pick = (idx, r)

    print("%d %d" % best_pick)


if __name__ == "__main__":
    main()

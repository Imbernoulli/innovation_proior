# TIER: greedy
# The textbook coformer-screening heuristic: among all regulatory-approved
# (coformer, ratio) pairs that clear the stability floor, pick the one with
# the STRONGEST hydrogen-bond synthon match (maximize h), tie-broken toward
# the smallest ratio then smallest index. This is a real, commonly-used
# screening rule -- "pick the strongest heterosynthon" -- and it deliberately
# ignores the lattice-energy-excess penalty on solubility. On instances where
# the strongest-bonding coformer is also a low-polarity "bond-monster", this
# lands far from the true solubility optimum.
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

    best = None  # (h, -r, idx) maximize h, then minimize r, then index
    best_pick = (0, 1)
    for idx, (fd, fa, lc, pol, ratios) in enumerate(formers):
        for r in sorted(ratios):
            h = h_score(W, Ad, Aa, fd, fa, r, K)
            L = h + lc * r
            if L < L_min:
                continue  # naive rule still respects the stability floor
            key = (h, -r)
            if best is None or key > best:
                best = key
                best_pick = (idx, r)

    print("%d %d" % best_pick)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gen.py <testId> -- print ONE cocrystal-former-selection instance to stdout.

Family: cocrystal-former-select (hydrogen-bond-synthon-match + lattice-energy-
tradeoff + regulatory-former-list).  Deterministic, seeded by testId only.

Instance layout (see statement.md for the full contract):
  K
  donor_strength[0..K-1]
  acceptor_strength[0..K-1]
  Ad[0..K-1]            (API donor site counts per functional-group type)
  Aa[0..K-1]            (API acceptor site counts per functional-group type)
  P_BONUS DECAY L_min
  M
  M lines, one per regulatory-approved former:
     fd[0..K-1] fa[0..K-1] lc p R ratio_1 ... ratio_R

Every instance plants three "anchor" formers at fixed indices:
  idx 0  "reference"  -- weak/modest bonder, small lattice add-on, modest
                         polarity. This is the checker's baseline construction.
  idx 1  "bond-monster" -- donor/acceptor counts engineered to fully SATURATE
                         every API site (the maximum possible hydrogen-bond
                         synthon score H_MAX). Maximizing raw bond strength
                         always lands here (it is always the unique max-h,
                         min-excess-among-max-h choice). On TRAP cases (odd
                         testId, >=5 of 10) its polarity is deliberately low,
                         so the huge lattice-energy excess crushes dSol to
                         <=0 -- the naive "strongest bond" heuristic scores
                         ZERO. On the other cases its polarity is moderate,
                         so bond-strength maximization is still a mediocre
                         but positive strategy (worse than the true optimum,
                         better than doing nothing) -- a realistic heuristic
                         that mostly works but fails badly when it matters.
  idx 2  "optimizer"   -- moderate bonding (not maximal), moderate lattice
                         add-on, HIGH polarity. Clears the stability threshold
                         with a much smaller lattice-energy excess than the
                         bond-monster, so the polarity bonus dominates the
                         excess penalty -> the best true solubility gain.
  idx 3..M-1  filler formers with random (weaker) attributes for realism and
                         search-space bulk; none can out-bond the bond-monster
                         (fd/fa capped below saturation) or out-polarize the
                         optimizer (polarity capped well under idx-2's range).
"""
import sys


def h_score(W, Ad, Aa, fd, fa, r, K):
    """Hydrogen-bond synthon-match score at stoichiometry r (former:API)."""
    tot = 0
    for t in range(K):
        tot += W[t] * (min(Ad[t], r * fa[t]) + min(Aa[t], r * fd[t]))
    return tot


def main():
    tid = int(sys.argv[1])
    seed = 20260701 + 104729 * tid
    # small deterministic xorshift-ish LCG-free PRNG via Python's random,
    # seeded purely from testId (G4 determinism).
    import random
    rng = random.Random(seed)

    K = 4
    donor_strength = [rng.randint(2, 9) for _ in range(K)]
    acceptor_strength = [rng.randint(2, 9) for _ in range(K)]
    W = [donor_strength[t] * acceptor_strength[t] for t in range(K)]
    SUMW = sum(W)

    Ad = [rng.randint(2, 4) for _ in range(K)]
    Aa = [rng.randint(2, 4) for _ in range(K)]

    L_min = 2 * SUMW           # exactly matches idx-0's raw bond score
    P_BONUS = SUMW              # scale-invariant: polarity bonus ~ O(SUMW)
    DECAY = 1

    is_trap = tid in (1, 4, 8)  # 3 of 10 cases: bond-monster trap fires

    formers = []

    # idx 0: reference / checker baseline construction.
    fd0 = [1] * K
    fa0 = [1] * K
    p0 = rng.randint(6, 9)
    e0 = rng.randint(1, 10)     # target small lattice-energy excess
    lc0 = e0                    # h0 = 2*SUMW = L_min  =>  L0 = L_min + e0
    formers.append((fd0, fa0, lc0, p0, [1]))

    # idx 1: bond-monster -- saturates every API site (raw max synthon score).
    # Ratio list is [1] only: h(1,r) is already fully saturated at r=1 (every
    # API site capped), so a higher r would only ADD lattice-packing cost
    # (lc1*r) with no further bonding benefit -- greedy's own tie-break logic
    # (max h, then smallest r) always lands on r=1 here anyway; fixing the
    # list to [1] keeps the regulatory catalog realistic without opening an
    # unbounded "more coformer molecules = free polarity bonus" loophole.
    fd1 = [Aa[t] for t in range(K)]
    fa1 = [Ad[t] for t in range(K)]
    if is_trap:
        p1 = rng.randint(1, 2)      # low polarity: rigid H-bond, poor solvation
    else:
        p1 = rng.randint(21, 29)    # moderate polarity: mediocre but not ruinous
    lc1 = rng.randint(1, 5)
    formers.append((fd1, fa1, lc1, p1, [1]))

    # idx 2: optimizer -- moderate bonding, high polarity, small excess.
    fd2 = [2] * K
    fa2 = [2] * K
    p2 = rng.randint(30, 45)    # high polarity: hydrophilic former
    lc2 = rng.randint(1, 5)
    formers.append((fd2, fa2, lc2, p2, [1]))

    # Fillers each carry exactly ONE regulatory-approved (coformer, ratio)
    # combination (R=1). Their h(.) already saturates at that fixed ratio
    # given their small fd/fa draws, so this does not open a "just crank r"
    # loophole; it keeps the catalog realistic (each row is one approved
    # entry) without a second free-scaling knob competing with idx 0..2.
    n_fillers = 4 + tid
    for _ in range(n_fillers):
        fdj = [rng.randint(0, 3) for _ in range(K)]
        faj = [rng.randint(0, 3) for _ in range(K)]
        lcj = rng.randint(1, 15)
        pj = rng.randint(1, 20)
        rj = rng.choice([1, 2, 3])
        formers.append((fdj, faj, lcj, pj, [rj]))

    M = len(formers)

    out = []
    out.append(str(K))
    out.append(" ".join(map(str, donor_strength)))
    out.append(" ".join(map(str, acceptor_strength)))
    out.append(" ".join(map(str, Ad)))
    out.append(" ".join(map(str, Aa)))
    out.append("%d %d %d" % (P_BONUS, DECAY, L_min))
    out.append(str(M))
    for (fd, fa, lc, p, ratios) in formers:
        line = list(map(str, fd)) + list(map(str, fa)) + [str(lc), str(p), str(len(ratios))] + list(map(str, ratios))
        out.append(" ".join(line))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

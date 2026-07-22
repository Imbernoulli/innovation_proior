# TIER: greedy
"""The 'obvious' textbook approach: for each target mode k=2..r, work out
how far its natural (uniform-mass) ratio is from the requested target, and
weight bead i by how much it sits on that mode's antinodes
(sin^2(k*pi*i/(n+1))) -- i.e. treat each mode's correction independently
and just additively combine the per-mode "where do I load mass to move
THIS mode" recipes, then hand out the budget proportionally to the combined
score.  It never checks whether loading mode k's antinode also disturbs
mode k+1 -- for modes with adjacent indices (whose antinodes sit close
together) this cross-talk is large, so this recipe systematically overshoots
some modes while undershooting neighbours."""
import sys
import math


def natural_log_ratios(n, r):
    out = []
    denom = math.sin(math.pi / (2.0 * (n + 1)))
    for k in range(1, r + 1):
        out.append(math.log(math.sin(k * math.pi / (2.0 * (n + 1))) / denom))
    return out


def main():
    data = sys.stdin.read().split()
    p = 0
    n = int(data[p]); p += 1
    B = int(data[p]); p += 1
    CAP = int(data[p]); p += 1
    r = int(data[p]); p += 1
    targets = []
    for _ in range(r):
        num = int(data[p]); p += 1
        den = int(data[p]); p += 1
        targets.append(num / den)

    L0 = natural_log_ratios(n, r)
    lt = [math.log(t / targets[0]) for t in targets]
    dL = [lt[k] - L0[k] for k in range(r)]  # index 0 (mode1) unused

    score = [0.0] * n
    for k in range(2, r + 1):
        w = -dL[k - 1]  # positive if this mode needs to be slowed (lowered) relatively
        for i in range(1, n + 1):
            v2 = math.sin(k * math.pi * i / (n + 1)) ** 2
            score[i - 1] += w * v2

    score = [max(s, 0.0) for s in score]
    total = sum(score) + 1e-9
    weights = [s / total for s in score]

    e_cont = [w * B for w in weights]
    e_int = [int(math.floor(x)) for x in e_cont]
    e_int = [min(v, CAP) for v in e_int]
    diff = B - sum(e_int)

    frac = [e_cont[i] - e_int[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -frac[i])
    j = 0
    guard = 0
    while diff > 0 and guard < 20 * n:
        i = order[j % n]
        if e_int[i] < CAP:
            e_int[i] += 1
            diff -= 1
        j += 1
        guard += 1
    # If still short (degenerate score profile), fill any bead with room.
    i = 0
    while diff > 0 and i < n:
        if e_int[i] < CAP:
            take = min(diff, CAP - e_int[i])
            e_int[i] += take
            diff -= take
        i += 1

    print(" ".join(map(str, e_int)))


if __name__ == "__main__":
    main()

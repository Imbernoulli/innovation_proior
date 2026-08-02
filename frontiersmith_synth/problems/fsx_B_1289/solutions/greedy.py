# TIER: greedy
"""The 'obvious' recipe an average coder writes first:

  1. Split the module sequence into a fixed number of roughly EQUAL-COST
     tranches (textbook 'engineering convenience' staging -- nobody looks at
     which specific module is informative, they just balance dollars per
     stage). Number of tranches is a size-based rule of thumb, not tied to
     information content at all.
  2. At each checkpoint, use a simple vote on the signals seen so far:
     keep going unless a clear majority (margin >= 2) say "Bad".

This uses the real option (it DOES sometimes abandon) but is blind to WHICH
modules carry information -- an informative module gets diluted inside a big
equal-cost batch together with several uninformative ones, so its signal is
swamped by noise before the checkpoint is even reached.
"""
import sys


def equal_cost_boundaries(M, costs):
    K = 3 if M >= 5 else (2 if M >= 3 else 1)
    K = min(K, M)
    total = sum(costs)
    target = total / K
    b = []
    cum = 0
    stage_num = 1
    for j in range(1, M + 1):
        cum += costs[j - 1]
        if stage_num < K and cum >= target * stage_num:
            b.append(j)
            stage_num += 1
    if not b or b[-1] != M:
        b.append(M)
    out = []
    for x in b:
        if not out or x > out[-1]:
            out.append(x)
    if out[-1] != M:
        out[-1] = M
    return out


def margin_table(gk, margin=2):
    tab = []
    for h in range(1 << gk):
        ng = bin(h).count("1")
        nb = gk - ng
        tab.append(0 if (nb - ng) >= margin else 1)
    return tab


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    costs = [int(next(it)) for _ in range(M)]
    # remaining fields (acc, p, VG, VB, sigma, F, r) unused by this heuristic

    boundaries = equal_cost_boundaries(M, costs)
    K = len(boundaries)

    out = [str(K)]
    out.append(" ".join(str(x) for x in boundaries))
    g = [0] + boundaries
    for k in range(1, K):
        gk = g[k]
        tab = margin_table(gk, margin=2)
        out.append(" ".join(str(x) for x in tab))
    print("\n".join(out))


if __name__ == "__main__":
    main()

import sys
from fractions import Fraction

def solve(data):
    idx = 0
    n = data[idx]; idx += 1
    K = data[idx]; idx += 1   # max planks per group (group size cap)
    b = []
    w = []
    for i in range(n):
        b.append(data[idx]); idx += 1
        w.append(data[idx]); idx += 1

    # We partition 0..n-1 into contiguous groups, each group of length in [1, K].
    # Group [l..r] has B = sum b, W = sum w (W>0 since each w>=1).
    # A group is ADMISSIBLE iff B*B*1 ... constraint: |B| / W  >= ratio threshold p/q.
    #   i.e. the group's "average steepness" |B|/W must be at least p/q.
    #   Compared exactly by cross multiplication: |B|*q >= p*W.
    # Merit of an admissible group = B*B (square of total beauty), summed over all groups.
    # We must partition the WHOLE array (cover every plank) using only admissible groups,
    # maximizing total merit. If no valid partition exists, output the string "IMPOSSIBLE".
    p = data[idx]; idx += 1
    q = data[idx]; idx += 1

    # prefix sums
    pb = [0]*(n+1)
    pw = [0]*(n+1)
    for i in range(n):
        pb[i+1] = pb[i] + b[i]
        pw[i+1] = pw[i] + w[i]

    NEG = None  # represent -infinity / unreachable
    # dp[i] = best total merit for covering planks 0..i-1 (prefix of length i)
    dp = [NEG]*(n+1)
    dp[0] = 0
    for i in range(1, n+1):
        best = NEG
        # last group is [j..i-1], length i-j in [1,K]
        lo = max(0, i-K)
        for j in range(lo, i):
            if dp[j] is NEG:
                continue
            B = pb[i] - pb[j]
            W = pw[i] - pw[j]
            # admissible iff |B|*q >= p*W  (W>0 always)
            if abs(B)*q >= p*W:
                cand = dp[j] + B*B
                if best is NEG or cand > best:
                    best = cand
        dp[i] = best
    if dp[n] is NEG:
        return "IMPOSSIBLE"
    return str(dp[n])

def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))

main()

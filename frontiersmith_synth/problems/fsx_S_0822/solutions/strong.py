# TIER: strong
# The insight: regulatory REDUNDANCY beats a tighter fit. Duplicate every
# positional channel onto a PARALOG PAIR of genes (2t, 2t+1) that both mirror
# bit t. Losing gene 2t still leaves 2t+1 reporting the same bit correctly
# (and vice versa), so the position's binary address survives ANY single
# knockout completely intact -- only the combination of bits that can arise
# changes, never the information content. We enumerate exactly which codes
# arise in the wild type and under every single-gene knockout and pin each one
# to the correct target type; no two different positions ever produce the
# same code under the same scenario, so there is no collision to resolve.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]; L = inst["L"]; target = inst["target"]

G = 2 * T
Win = [[0] * T for _ in range(G)]
for t in range(T):
    Win[2 * t][t] = 1
    Win[2 * t + 1][t] = 1
W = [[0] * G for _ in range(G)]
bias = [0] * G

decode = [0] * (1 << G)


def code_for(p, ko):
    c = 0
    for t in range(T):
        xt = (p >> t) & 1
        if not xt:
            continue
        for gi in (2 * t, 2 * t + 1):
            if gi == ko:
                continue
            c |= (1 << gi)
    return c


for p in range(L):
    decode[code_for(p, None)] = target[p]
    for g in range(G):
        decode[code_for(p, g)] = target[p]

print(json.dumps({"G": G, "Win": Win, "W": W, "bias": bias, "decode": decode}))

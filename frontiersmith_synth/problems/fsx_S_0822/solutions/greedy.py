# TIER: greedy
# The obvious "just fit the target" network: one gene per positional bit, each
# gene simply mirrors that bit (Win = identity, no cross-regulation), so the
# final expression vector IS the position's binary address and decode[p] =
# target[p] reconstructs the wild type EXACTLY. It never occurred to this
# design that the network might lose a gene: knocking out gene i zeroes bit i
# for EVERY position, so every position whose true bit i was 1 silently reads
# out as the DIFFERENT position with bit i forced to 0 -- roughly half the
# body relabeled to the wrong segment's identity.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]; L = inst["L"]; target = inst["target"]

G = T
Win = [[1 if t == i else 0 for t in range(T)] for i in range(G)]
W = [[0] * G for _ in range(G)]
bias = [0] * G

decode = [0] * (1 << G)
for p in range(L):
    decode[p] = target[p]

print(json.dumps({"G": G, "Win": Win, "W": W, "bias": bias, "decode": decode}))

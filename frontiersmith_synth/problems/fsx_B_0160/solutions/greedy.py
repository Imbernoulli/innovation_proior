# TIER: greedy
# "Quote the longest lead time everywhere and let the stations absorb it." Top-down,
# each node pushes its outbound service time as high as feasible: S_i = SI_i + T_i so
# tau_i = 0 (NO safety stock at that node). Internal nodes therefore carry nothing,
# but the stations are capped at s_max, so all of the accumulated lead time piles onto
# the stations' net replenishment time -- and stations have the HIGHEST holding cost.
# This over-concentrates the most expensive safety stock at the leaves: feasible, but
# well worse than a balanced placement (typically below the decoupled baseline too).
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; parent = inst["parent"]; T = inst["T"]; smax = inst["smax"]
Sext = inst["Sext"]; level = inst["level"]
S = [0] * n
for i in sorted(range(n), key=lambda i: level[i]):   # parents before children
    SI = Sext if parent[i] < 0 else S[parent[i]]
    S[i] = min(SI + T[i], smax[i])
print(json.dumps({"S": S}))

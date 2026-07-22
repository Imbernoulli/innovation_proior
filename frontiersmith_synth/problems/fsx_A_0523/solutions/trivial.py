# TIER: trivial
# All boilers online for the whole horizon, each carrying a capacity-proportional
# share of demand (raised to its minimum output when the share falls below it).
# This reproduces the checker's reference construction -> ratio ~= 0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    i = 0
    T = int(toks[i]); i += 1
    K = int(toks[i]); i += 1
    D = [float(toks[i + j]) for j in range(T)]; i += T
    caps = []; pmins = []
    for _ in range(K):
        caps.append(float(toks[i])); pmins.append(float(toks[i + 1])); i += 9
    total = sum(caps)
    out = []
    for t in range(T):
        row = []
        for k in range(K):
            share = D[t] * caps[k] / total
            o = share if share > pmins[k] else pmins[k]
            row.append("%.6f" % o)
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


main()

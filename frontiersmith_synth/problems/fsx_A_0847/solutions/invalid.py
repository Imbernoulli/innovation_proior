# TIER: invalid
"""Emits a structurally well-formed but INFEASIBLE artifact: the state-encoding
table maps every state to the SAME code (not a bijection). The checker must
reject this deterministically -> Ratio 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    S = int(next(it)); K = int(next(it))
    for _ in range(S * K):
        next(it)

    b = (S - 1).bit_length()
    m = (K - 1).bit_length()

    out = []
    out.append("%d %d" % (b, m))
    out.append(str(S))
    for s in range(S):
        out.append("%d 0" % s)           # every state -> code 0 (not a bijection)
    out.append("0")                       # zero gates
    outwires = [0 for _ in range(b)]
    out.append(" ".join(str(w) for w in outwires))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

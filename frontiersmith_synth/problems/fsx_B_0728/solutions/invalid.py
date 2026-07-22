# TIER: invalid
# Emits well-formed tokens with the right counts but ignores the vertex-group parity
# floor entirely (always Valley) -- must be rejected by the checker's feasibility gate.
import sys


def main():
    text = sys.stdin.read()
    toks = text.split()
    it = iter(toks)
    N = int(next(it)); K = int(next(it)); G = int(next(it))
    for _ in range(N):
        next(it)
    for _ in range(N - 1):
        next(it)
    for _ in range(K):
        next(it); next(it); next(it)
    for _ in range(G):
        for _ in range(5):
            next(it)

    step_out = ['V'] * (N - 1)
    hinge_out = ['V'] * K
    heights = list(range(N))
    out = []
    out.append(" ".join(step_out))
    out.append(" ".join(hinge_out))
    out.append(" ".join(map(str, heights)))
    print("\n".join(out))


if __name__ == "__main__":
    main()

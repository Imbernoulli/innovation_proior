# TIER: trivial
"""Reproduces the checker's own baseline: recompute every output's subtree
completely from leaves, independently, with zero sharing awareness and zero
use of DUP/SWAP/OVER/ROT/STORE/LOAD. This is the textbook "just walk the
expression tree" codegen -- ignores the DAG's sharing entirely."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    for _ in range(N):
        next(it)  # leaf values (unused: we only need structure)
    nodes = {}
    for i in range(1, M + 1):
        op = next(it); cl = next(it); cr = next(it)
        nodes[i] = (op, cl, cr)
    outs = [next(it) for _ in range(K)]

    lines = []

    def emit(ref):
        if ref[0] == "L":
            lines.append("PUSH %d" % int(ref[1:]))
        else:
            idx = int(ref[1:])
            op, cl, cr = nodes[idx]
            emit(cl)
            emit(cr)
            lines.append("OP %s" % op)

    sys.setrecursionlimit(1000000)
    for o in outs:
        emit(o)
        lines.append("OUTPUT")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

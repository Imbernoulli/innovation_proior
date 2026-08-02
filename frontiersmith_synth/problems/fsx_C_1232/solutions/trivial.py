# TIER: trivial
"""
Reproduces the checker's own internal baseline B exactly: touch ONLY the
types that need no dependency reasoning at all (deps[t] == []) -- create
each one, apply whichever single opcode looks best from state 0. No
attempt at any multi-step chain, no attempt to satisfy any type that has a
prerequisite.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); C = int(next(it)); budget = int(next(it))
    deps = [None] * T
    tables = [None] * T
    for t in range(T):
        k = int(next(it))
        d = [int(next(it)) for _ in range(k)]
        S = int(next(it))
        table = [[int(next(it)) for _ in range(C)] for _ in range(S)]
        deps[t] = d
        tables[t] = table

    calls = []
    for t in range(T):
        if deps[t] or len(calls) + 2 > budget:
            continue
        best_c, best_v = 0, -1
        for c in range(C):
            ns = tables[t][0][c]
            v = ns * ns if ns >= 1 else 0
            if v > best_v:
                best_v, best_c = v, c
        line = len(calls)
        calls.append(f"C {t}")
        calls.append(f"O {line} {best_c}")

    out = [str(len(calls))] + calls
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

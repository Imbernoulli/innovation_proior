# TIER: trivial
"""Reproduce the checker's internal reference manifest: signatures with d[0]==0 and all
other coordinates in {0,1}, minus the protected ones. Scores ~0.1."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    pw = [3 ** i for i in range(n)]
    forb = set()
    for _ in range(m):
        code = 0
        for i in range(n):
            code += int(next(it)) * pw[i]
        forb.add(code)

    out = []
    for mask in range(1 << (n - 1)):
        digs = [0] * n
        code = 0
        for j in range(n - 1):
            if (mask >> j) & 1:
                digs[j + 1] = 1
                code += pw[j + 1]
        if code not in forb:
            out.append(" ".join(str(x) for x in digs))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()

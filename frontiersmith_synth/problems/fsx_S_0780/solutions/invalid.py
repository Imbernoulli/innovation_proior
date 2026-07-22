# TIER: invalid
# Copies the corpus like trivial.py, but corrupts program 0's final instruction
# (forces it to "ADD t0 t0") so its computed value almost never matches the
# original -> the equivalence gate rejects it -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); P = int(next(it)); SEED = int(next(it))
    out = ["MACROS 0"]
    progs_lines = []
    for k in range(N):
        assert next(it) == "PROGRAM"
        kk = int(next(it)); L = int(next(it))
        lines = []
        for _ in range(L):
            op = next(it); a = next(it); b = next(it)
            lines.append("%s %s %s" % (op, a, b))
        assert next(it) == "OUT"
        out_tok = next(it)
        if kk == 0 and L >= 1:
            lines[-1] = "ADD t0 t0"
        progs_lines.append((kk, L, lines, out_tok))

    out.append("PROGRAMS %d" % N)
    for (kk, L, lines, out_tok) in progs_lines:
        out.append("PROGRAM %d %d" % (kk, L))
        out.extend(lines)
        out.append("OUT %s" % out_tok)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

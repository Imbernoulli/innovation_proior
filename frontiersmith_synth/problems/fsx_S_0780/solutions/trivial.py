# TIER: trivial
# No compression at all: re-emit the corpus verbatim with zero macros.
# F == B exactly -> this reproduces the checker's own baseline (ratio == 0.1).
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
        progs_lines.append((kk, L, lines, out_tok))

    out.append("PROGRAMS %d" % N)
    for (kk, L, lines, out_tok) in progs_lines:
        out.append("PROGRAM %d %d" % (kk, L))
        out.extend(lines)
        out.append("OUT %s" % out_tok)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

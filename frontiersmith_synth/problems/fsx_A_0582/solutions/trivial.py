# TIER: trivial
# Do nothing clever: re-emit the bureaucrat's ritual verbatim.  Same op count as
# the baseline -> scores the calibrated ~0.10.
import sys

NIN = 8

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    p = int(next(it)); L = int(next(it))
    prog = []
    for _ in range(L):
        op = next(it)
        if op == "const":
            prog.append("const %s" % next(it))
        else:
            prog.append("%s %s %s" % (op, next(it), next(it)))
    out = [str(L)] + prog
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()

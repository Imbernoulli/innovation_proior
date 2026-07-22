# TIER: invalid
# Inks exactly the given anchors as raw top-level DOTs, with no symmetrization
# at all. Covers every anchor but is (generically) not invariant under the 8
# symmetries -> the checker's invariance gate scores this 0 on every case.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); A = int(next(it))
    anchors = [(int(next(it)), int(next(it))) for _ in range(A)]

    out = ["DOT %d %d" % (x, y) for (x, y) in anchors]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

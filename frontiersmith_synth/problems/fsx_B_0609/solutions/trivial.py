# TIER: trivial
# Packed-at-the-front layout: keys occupy cells 0..N-1, all gaps at the tail.
# This is exactly the checker's baseline construction -> scores ~0.1.
import sys


def main():
    d = sys.stdin.buffer.read().split()
    N = int(d[0])
    out = ["%d 0" % N, " ".join(str(i) for i in range(N))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

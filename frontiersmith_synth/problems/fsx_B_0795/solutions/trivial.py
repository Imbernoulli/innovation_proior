# TIER: trivial
# Stock every target as its own dedicated table entry: m=T, one factor per target.
# This reproduces the checker's own baseline construction exactly.
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0
    p = int(data[pos]); pos += 1
    g = int(data[pos]); pos += 1
    LAMBDA = int(data[pos]); pos += 1
    T = int(data[pos]); pos += 1
    targets = [int(x) for x in data[pos:pos + T]]

    out = []
    out.append(str(T))
    out.append(" ".join(str(t) for t in targets))
    for i in range(T):
        out.append("1 %d 1" % (i + 1))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

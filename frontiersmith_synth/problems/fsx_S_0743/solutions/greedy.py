# TIER: greedy
# The obvious "use the macro mechanism" attempt: put every GIVEN anchor into one
# macro (no attempt to notice that several anchors are just other images of the
# same underlying orbit under the 8 symmetries), then call that macro under all
# 8 symmetries. Always feasible and already far cheaper than the literal
# baseline -- but pays once per given anchor even when many are redundant.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); A = int(next(it))
    anchors = [(int(next(it)), int(next(it))) for _ in range(A)]

    out = ["DEF 0"]
    for (x, y) in anchors:
        out.append("DOT %d %d" % (x, y))
    out.append("END")
    for g in range(8):
        out.append("CALL 0 %d" % g)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

# TIER: invalid
# Emits a dispatch that grossly exceeds each unit's capacity -- must score 0.
import sys


def main():
    it = iter(sys.stdin.read().split())
    N = int(next(it)); T = int(next(it))
    caps = []
    for _ in range(N):
        caps.append(int(next(it)))
        next(it); next(it); next(it); next(it)  # skip m,a,b,fast
    # skip J and D entirely -- not needed for garbage output

    out_lines = []
    for _t in range(T):
        row = [caps[i] * 9.0 + 5.0 for i in range(N)]
        out_lines.append(" ".join("%.6f" % x for x in row))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

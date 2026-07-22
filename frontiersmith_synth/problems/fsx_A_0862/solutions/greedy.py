# TIER: greedy
# The obvious first idea: since a single arm alone may use up to
# floor(sqrt(P)) speed without ever risking the shared cap, just run every
# arm ALONE at that solo max speed, one arm at a time, in input order,
# until it is done, then move to the next arm. This never touches the cap
# and always finishes -- but it fully serializes every arm (no overlap
# between the long "critical" chain and the many short ones) and, because
# power is quadratic, running fast for a fixed distance burns energy
# proportional to speed itself, so blindly always maxing out speed also
# wastes energy on arms that had plenty of slack to go slower.
import math
import sys


def main():
    data = sys.stdin.read().split("\n")
    K, P, A = (int(x) for x in data[0].split())
    D = [int(x) for x in data[1].split()]

    vmax = math.isqrt(P)
    if vmax < 1:
        vmax = 1

    out = []
    for i in range(K):
        remaining = D[i]
        while remaining > 0:
            v = min(vmax, remaining)
            row = [0] * K
            row[i] = v
            out.append(" ".join(map(str, row)))
            remaining -= v
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()

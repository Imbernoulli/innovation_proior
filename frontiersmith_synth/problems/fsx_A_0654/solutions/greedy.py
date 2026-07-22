# TIER: greedy
# The "obvious" textbook approach: notice each rotating recipe block is banded
# and skip its zero entries every single day -- a real, local optimization --
# but simulate day by day regardless of how large T is. It never notices that
# the day-to-day recipe schedule repeats with period p, so it stays linear in
# T even when a handful of recalibration days are the only irregularity across
# thousands of otherwise-identical rotations.
import sys


def parse(tokens):
    it = iter(tokens)

    def nx():
        return int(next(it))

    S = nx(); K = nx(); T = nx(); p = nx(); m = nx()
    for _ in range(S):
        nx()  # x0
    for _k in range(K):
        for _i in range(S):
            for _j in range(S):
                nx()
        for _i in range(S):
            nx()
    pattern = [nx() for _ in range(p)]
    overrides = []
    for _o in range(m):
        t = nx()
        for _i in range(S):
            for _j in range(S):
                nx()
        for _i in range(S):
            nx()
        overrides.append(t)
    return S, K, T, p, m, pattern, overrides


def main():
    data = sys.stdin.read().split()
    S, K, T, p, m, pattern, overrides = parse(data)
    ov_idx = {t: i for i, t in enumerate(overrides)}

    actions = []
    for t in range(T):
        if t in ov_idx:
            actions.append(("STEP", "O", ov_idx[t]))
        else:
            k = pattern[t % p]
            actions.append(("BAND", k))

    lines = [str(len(actions))]
    for a in actions:
        lines.append(" ".join(str(x) for x in a))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

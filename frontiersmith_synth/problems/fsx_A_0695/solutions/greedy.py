# TIER: greedy
"""The obvious first attempt: robustify against a SINGLE representative scenario --
the one with the largest total inflow over the whole horizon -- and plan a release
calendar that is optimal (envelope-minimal) FOR THAT ONE SCENARIO ALONE. This is exactly
"a single-scenario optimum": it looks rigorous (it does hold water back to protect a
worst case) but it is blind to the fact that a DIFFERENT scenario can be wetter at an
EARLIER week even though its grand total is smaller -- the running max has to be taken
over ALL scenarios at EVERY prefix, not just picked once by final total."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); K = int(next(it)); C = int(next(it)); D = int(next(it))
    S0 = int(next(it)); Rmax = int(next(it))
    next(it); next(it)
    scenarios = [[int(next(it)) for _ in range(T)] for _ in range(K)]

    kstar = max(range(K), key=lambda k: sum(scenarios[k]))
    sc = scenarios[kstar]
    cum = [0] * (T + 1)
    for t in range(1, T + 1):
        cum[t] = cum[t - 1] + sc[t - 1]

    Rel = [0] * (T + 1)
    for t in range(1, T + 1):
        want = max(0, S0 + cum[t] - C)
        Rel[t] = min(max(want, Rel[t - 1]), Rel[t - 1] + Rmax)

    r = [Rel[t] - Rel[t - 1] for t in range(1, T + 1)]
    print(" ".join(str(x) for x in r))


if __name__ == "__main__":
    main()

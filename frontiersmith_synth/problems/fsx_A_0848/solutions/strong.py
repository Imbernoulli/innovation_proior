# TIER: strong
"""The insight: this is NOT a curve-fitting problem, it is a small,
exactly-enumerable HYPOTHESIS SPACE of deterministic counter automata.
There are only 4! role permutations x (U_MAX-U_MIN+1) magnitudes x 2
starting polarities candidate wirings. Rather than regressing a
continuous, order-blind surrogate against symbol counts (which throws
away all sequencing information the persistent polarity flag depends
on), SIMULATE every candidate exactly on every logged training burst
(honoring the polarity flag faithfully, character by character) and
score each candidate by total absolute deviation from the logged
(possibly noisy) readings. Because a wrong role assignment or magnitude
accumulates large, systematic error on any training row that contains a
polarity pulse, the minimum-total-error candidate recovers the true
wiring with overwhelming reliability from just a few dozen logged bursts.
Since the recovered automaton is an exact simulator (not a curve fit),
it extrapolates correctly to bursts of any length, including the long,
adversarially role-clustered held-out ones."""
import sys
from itertools import permutations

ALPHABET = "ABCD"
U_LO, U_HI = 1, 12  # search a bit wider than the true [2,9] range, cheaply


def simulate(s, inc, dec, flip, noop, u, p0):
    c = 0
    p = p0
    for ch in s:
        if ch == inc:
            c += p * u
        elif ch == dec:
            c -= p * u
        elif ch == flip:
            p = -p
    return c


def main():
    header = sys.stdin.readline().split()
    K = int(header[1])
    rows = []
    for _ in range(K):
        parts = sys.stdin.readline().split()
        s, y = parts[0], int(parts[1])
        rows.append((s, y))

    best = None
    best_err = None
    for inc, dec, flip, noop in permutations(ALPHABET):
        for u in range(U_LO, U_HI + 1):
            for p0 in (1, -1):
                err = 0
                for s, y in rows:
                    pred = simulate(s, inc, dec, flip, noop, u, p0)
                    err += abs(pred - y)
                    if best_err is not None and err >= best_err:
                        break
                if best_err is None or err < best_err:
                    best_err = err
                    best = (inc, dec, flip, noop, u, p0)

    inc, dec, flip, noop, u, p0 = best
    print(inc, dec, flip, noop, u, p0)


if __name__ == "__main__":
    main()

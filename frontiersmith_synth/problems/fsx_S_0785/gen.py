#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE ghost-train arrival log to stdout.

Theme: a railway dispatcher is trying to learn the timetable of a "ghost
train" that keeps visiting platform numbers according to a hidden law. Each
testId is a DIFFERENT hidden timetable:

  platform(t) = B0 + B1*r + B2*r^2 + S0*(t mod P) + jitter(t),   r = t // P

  - S0*(t mod P) is the FAST inner shunting pattern within one "timetable
    epoch" of length P (nested stride).
  - B0 + B1*r + B2*r^2 is the SLOW epoch-level drift: which platform the
    epoch as a whole is anchored to, as a function of the epoch counter r
    (a modular/counter-driven regime switch: it changes only when r changes).
  - jitter(t) is a small bounded dispatcher fudge factor, deterministic in
    t but built from bit-mixing (xor/shift) that is NOT expressible in the
    solver's arithmetic grammar (+ - * // % **), so it can never be nailed
    exactly -- it is the irreducible noise floor.

The log printed here covers epochs r = 0 .. Rtrain-1 ONLY. The grader's
held-out queries live at epochs r >= Rtrain (never seen here) -- genuine
extrapolation, not interpolation.

STDOUT: header "N P testId", then N lines, one platform number per tick
t = 0 .. N-1. No hidden coefficients are printed -- data rows only.
"""
import sys


def hidden_params(tid):
    """Hidden timetable for this test id (identical copy lives in verify.py)."""
    import random
    rng = random.Random(5170001 + tid * 104729)
    P = rng.randint(50, 110)
    S0 = rng.randint(3, 9)
    B0 = rng.randint(5, 60)
    B1 = rng.randint(2, 6)
    B2 = 0 if tid <= 3 else rng.randint(1, 3)
    J3 = rng.choice([3, 4, 5, 6])
    Jc = rng.randint(0, 999983)
    Rtrain = 20 + 6 * tid
    return P, S0, B0, B1, B2, J3, Jc, Rtrain


def jitter(t, J3, Jc):
    h = (t * 2654435761 + Jc) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= (h >> 15)
    return h % J3


def platform(t, P, S0, B0, B1, B2, J3, Jc):
    r = t // P
    return B0 + B1 * r + B2 * r * r + S0 * (t % P) + jitter(t, J3, Jc)


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    P, S0, B0, B1, B2, J3, Jc, Rtrain = hidden_params(tid)
    N = P * Rtrain
    lines = ["%d %d %d" % (N, P, tid)]
    for t in range(N):
        lines.append(str(platform(t, P, S0, B0, B1, B2, J3, Jc)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

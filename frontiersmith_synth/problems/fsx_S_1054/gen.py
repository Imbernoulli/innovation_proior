#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE island-microgrid tariff instance to stdout.
Deterministic: all randomness is seeded from testId only.
"""
import random
import sys

T = 24
BASE_SHAPE = [2, 2, 1, 1, 1, 2, 4, 6, 5, 4, 4, 4,
              5, 5, 4, 4, 5, 7, 9, 10, 9, 7, 5, 3]

P_MIN, P_MAX = 0.05, 0.65
ALPHA = 0.0001
TOL_FRAC = 0.30
EPS_MOD = 300

# ladder: (N, need_lo, need_hi, rate_lo, rate_hi) -- difficulty grows with testId.
# need >> rate everywhere: every household's daily charge session spans several
# hours, so a homogeneous fleet's response genuinely depends on which hours the
# published algorithm ranks first, not just "everyone grabs one cheap slot".
LADDER = [
    (10,  200, 360,  60, 120),
    (20,  240, 400,  80, 140),
    (35,  280, 480,  80, 160),
    (55,  320, 560, 100, 180),
    (60,  360, 600, 100, 200),
    (90,  400, 680, 120, 220),
    (110, 440, 720, 120, 240),
    (150, 480, 800, 140, 260),
    (210, 520, 880, 140, 280),
    (260, 560, 960, 160, 300),
]


def gen(test_id):
    N, nlo, nhi, rlo, rhi = LADDER[test_id - 1]
    rnd = random.Random(test_id * 1000003 + 999)
    lines = []
    lines.append(str(N))
    lines.append(f"{P_MIN} {P_MAX} {ALPHA} {TOL_FRAC} {EPS_MOD}")
    for i in range(N):
        scale = rnd.uniform(0.7, 1.3)
        row = []
        for t in range(T):
            noise = rnd.uniform(-0.5, 0.5)
            v = max(1, round(BASE_SHAPE[t] * scale + noise))
            row.append(v)
        r = rnd.randint(rlo, rhi)
        nd = rnd.randint(max(nlo, r + 1), max(nlo, r + 1, nhi))
        row.append(nd)
        row.append(r)
        lines.append(" ".join(str(x) for x in row))
    return "\n".join(lines) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(gen(test_id))


if __name__ == "__main__":
    main()

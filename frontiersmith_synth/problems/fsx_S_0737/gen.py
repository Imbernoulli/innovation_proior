#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE noisy snapshot log of a hidden 1D binary lantern
row to stdout.

A row of W lanterns (0=dark,1=lit), arranged in a CIRCLE (wraps around), is
driven by a hidden LOCAL RULE: every tick, lantern i's new state depends only
on lanterns i-1, i, i+1 (radius 1), applied to all lanterns simultaneously.
Each testId fixes a DIFFERENT hidden rule (one of 256 possible radius-1 rules).

You never get to watch every tick. You get a logbook of SPARSE, IRREGULAR
snapshots (gaps of >=2 ticks between reads -- never two consecutive ticks),
and each reading is itself corrupted: the night watchman misreads a small
percent of lanterns (a lit lantern read dark or vice versa) every time he
looks, because they flicker.

STDOUT prints ONLY: a header "<testId> <W> <m>", then m lines
"<tick> <row-as-W-char 0/1 string>". The hidden rule number, the noise seed,
and the held-out grading trace are NEVER printed -- they live only inside the
checker (verify.py), reconstructed purely from testId.
"""
import sys

# 10 curated, non-degenerate elementary radius-1 rules (every rule genuinely
# depends on cL, cM AND cR -- no 0/1/2-variable reduction), empirically chosen
# so a same-step-pretend majority-vote fit tracks SOME of them tolerably and
# gets badly aliased by others once the gap composition kicks in -- a mix of
# mild and hard cases, not a monotone difficulty ramp. One rule per testId.
RULES = [142, 227, 194, 24, 130, 190, 152, 97, 159, 43]


def _build(t):
    """Deterministic instance builder shared VERBATIM with verify.py. Every
    random draw below (gaps, initial rows, both noise passes, the held-out
    initial condition) comes from ONE seeded stream in a fixed order, so
    gen.py and verify.py -- given the same testId -- reproduce identical
    hidden state without ever writing it to disk."""
    import random
    rulenum = RULES[(t - 1) % len(RULES)]
    W = 80 + 12 * (t - 1)
    m = 5
    rng = random.Random(900001 + t * 7919)
    maxgap = 3 + (t - 1) // 2
    gaps = [rng.randint(2, maxgap) for _ in range(m - 1)]
    times = [0]
    for g in gaps:
        times.append(times[-1] + g)
    p_train = 0.02 + 0.003 * (t - 1)
    p_final = 0.18
    L = 500
    table = [(rulenum >> idx) & 1 for idx in range(8)]

    def step(row):
        Wn = len(row)
        return [table[row[(i - 1) % Wn] * 4 + row[i] * 2 + row[(i + 1) % Wn]] for i in range(Wn)]

    def noisy(row, p):
        return [(1 - b) if rng.random() < p else b for b in row]

    train_row0 = [rng.randint(0, 1) for _ in range(W)]
    clean = train_row0[:]
    snapshots = [noisy(clean, p_train)]
    for g in gaps:
        for _ in range(g):
            clean = step(clean)
        snapshots.append(noisy(clean, p_train))

    grade_row0 = [rng.randint(0, 1) for _ in range(W)]
    g_true = grade_row0[:]
    for _ in range(L):
        g_true = step(g_true)
    g_observed_final = noisy(g_true, p_final)

    return dict(rulenum=rulenum, W=W, times=times, snapshots=snapshots,
                p_train=p_train, p_final=p_final, L=L,
                grade_row0=grade_row0, g_observed_final=g_observed_final,
                table=table)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    d = _build(t)
    W, times, snaps = d["W"], d["times"], d["snapshots"]
    out = ["%d %d %d" % (t, W, len(times))]
    for ti, row in zip(times, snaps):
        out.append("%d %s" % (ti, "".join(str(b) for b in row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

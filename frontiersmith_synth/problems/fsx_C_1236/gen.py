#!/usr/bin/env python3
"""gen.py <testId> -- prints one circuit-breaker-tuning instance to stdout.

Deterministic: testId fully determines the instance (an internal RNG is
seeded from `1000 + testId` only). The instance is a fixed-length, already
decided timeline of "would this call have succeeded" bits for a single
downstream dependency; the participant tunes breaker parameters against it
(the checker replays those parameters causally, in order, against this same
timeline -- there is no hidden/held-out data).

Instance shape:
    line 1: "T R CF"     T ticks (1-indexed 1..T), R = reward per successful
                          call, CF = cost per failed call
    line 2: "o_1 ... o_T"  o_t in {0,1}: whether a call attempted at tick t
                          would have succeeded

The timeline is built by concatenating REGIMES:
    HEALTHY   -- p(success) = 0.97 (occasional blips only)
    HARD_DOWN -- p(success) = 0.02 (a clean, unambiguous outage)
    GRAY      -- a bursty, positively-autocorrelated gray-failure regime:
                 it alternates between short GOOD sub-bursts (p=0.93) and
                 BAD sub-bursts (p=0.07), each sub-burst 6-9 ticks long. Any
                 short window sampled inside one sub-burst looks confidently
                 healthy (or confidently dead) -- but the regime as a whole
                 is not recovering: it keeps flipping. Averaged over any
                 window spanning several sub-bursts, success sits near 50%.
Per-testId regime plans are fixed (only the coin flips inside each regime,
and each GOOD/BAD sub-burst length, are randomized), so every instance
deterministically contains the intended mix: clean outages, recurring
outages with DIFFERENT recovery durations, and bursty gray-failure
stretches long enough to span several sub-bursts. Healthy padding around
each bad stretch is sized (per testId, as a fixed function of that
stretch's own severity) so the "always call" reference stays a robust,
non-degenerate positive baseline.
"""
import sys, random

P_HEALTHY = 0.97
P_HARD = 0.02
GOOD_P = 0.93
BAD_P = 0.07
RUN_LO, RUN_HI = 6, 9
R_REWARD = 1
CF_COST = 3

# Per testId: (regime, duration) structure. 'H' segments carry a weight (all
# 1 here); the actual healthy durations are derived below from each test's
# total bad-segment severity so every instance's "always call" baseline
# lands in a robust, comparable, strictly-positive range.
STRUCTURES = {
    1: [('H', 1), ('D', 5), ('H', 1)],
    2: [('H', 1), ('D', 15), ('H', 1)],
    3: [('H', 1), ('G', 70), ('H', 1)],
    4: [('H', 1), ('D', 4), ('H', 1), ('D', 4), ('H', 1), ('D', 4), ('H', 1)],
    5: [('H', 1), ('G', 100), ('H', 1)],
    6: [('H', 1), ('D', 10), ('H', 1), ('D', 40), ('H', 1)],
    7: [('H', 1), ('D', 8), ('H', 1), ('D', 20), ('H', 1), ('D', 45), ('H', 1)],
    8: [('H', 1), ('G', 60), ('H', 1), ('D', 15), ('H', 1), ('G', 70), ('H', 1)],
    9: [('H', 1), ('D', 10), ('H', 1), ('G', 70), ('H', 1), ('D', 30), ('H', 1),
        ('G', 60), ('H', 1)],
    10: [('H', 1), ('G', 35), ('H', 1), ('G', 35), ('H', 1), ('G', 35), ('H', 1),
         ('D', 30), ('H', 1)],
}

ALPHA_B = 1.2  # healthy padding = ALPHA_B * (bad segments' expected cost magnitude)


def _avg_contrib(p_success, cf):
    return p_success * R_REWARD - (1.0 - p_success) * cf


def build_plan(struct):
    avg_h = _avg_contrib(P_HEALTHY, CF_COST)
    avg_d = _avg_contrib(P_HARD, CF_COST)
    avg_g = _avg_contrib(0.5, CF_COST)  # gray averages to ~50% over full bursts
    d_total = sum(d for r, d in struct if r == 'D')
    g_total = sum(d for r, d in struct if r == 'G')
    h_weight_total = sum(w for r, w in struct if r == 'H')
    bad_magnitude = -(avg_d * d_total + avg_g * g_total)
    needed_h = bad_magnitude * ALPHA_B / avg_h
    if needed_h < h_weight_total * 8:
        needed_h = h_weight_total * 8
    plan = []
    for r, val in struct:
        if r == 'H':
            dur = max(8, round(needed_h * val / h_weight_total))
            plan.append(('H', dur))
        else:
            plan.append((r, val))
    return plan


def gray_seq(rng, dur):
    seq = []
    cur = 'GOOD'
    remaining = rng.randint(RUN_LO, RUN_HI)
    for _ in range(dur):
        if remaining == 0:
            cur = 'BAD' if cur == 'GOOD' else 'GOOD'
            remaining = rng.randint(RUN_LO, RUN_HI)
        p = GOOD_P if cur == 'GOOD' else BAD_P
        seq.append(1 if rng.random() < p else 0)
        remaining -= 1
    return seq


def build(test_id: int):
    rng = random.Random(1000 + test_id)
    plan = build_plan(STRUCTURES[test_id])
    outcomes = []
    p_of = {'H': P_HEALTHY, 'D': P_HARD}
    for regime, dur in plan:
        if regime == 'G':
            outcomes += gray_seq(rng, dur)
        else:
            p = p_of[regime]
            for _ in range(dur):
                outcomes.append(1 if rng.random() < p else 0)

    # Safety net (deterministic, only ever needed on a pathological draw):
    # guarantee the "always call" baseline the checker uses is strictly
    # positive, by topping up with extra healthy ticks if an unlucky draw
    # left it non-positive.
    def naive_total(seq):
        return sum(R_REWARD if o else -CF_COST for o in seq)
    attempt = 0
    while naive_total(outcomes) <= 0 and attempt < 400:
        outcomes.append(1 if rng.random() < P_HEALTHY else 0)
        attempt += 1
    return outcomes


def main():
    test_id = int(sys.argv[1])
    outcomes = build(test_id)
    T = len(outcomes)
    out = [f"{T} {R_REWARD} {CF_COST}", " ".join(str(o) for o in outcomes)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

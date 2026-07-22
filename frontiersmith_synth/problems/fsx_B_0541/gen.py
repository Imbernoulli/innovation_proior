import sys, random


def gen(t):
    # Deterministic instance for difficulty rung t (1..10). Seed from t only.
    rng = random.Random(90000 + 7919 * t)

    # ----- scale ladder -----
    N = 14 + 3 * t                       # 17 .. 44 jobs
    # fraction of jobs that are soft "dressing" jobs (negative wear delta)
    frac_soft = 0.38 + 0.02 * ((t % 3))  # ~0.38..0.42
    n_soft = max(3, int(round(frac_soft * N)))
    n_hard = N - n_soft

    # Re-dress cost. Big enough that resetting before every job is bad, small
    # enough that a few resets can be worthwhile when soft capacity runs out.
    T_r = 22 + 2 * (t % 4)               # 22 .. 28

    # A fully worn wheel is as bad as it gets: wear saturates at W_max (and never
    # drops below 0). This bounds the (1+w)^2 penalty so the score does not
    # saturate, while keeping the wear-control problem sharp.
    W_max = 4

    jobs = []  # (size, delta)

    # ---- HARD (production) jobs: LARGE size, positive wear delta ----
    # sizes are big so the (1+wear)^2 amplification bites hard; deltas are the
    # damage they inflict on the wheel.
    for _ in range(n_hard):
        s = rng.randint(5, 11)
        d = rng.choice([1, 1, 2, 2, 2, 3])
        jobs.append((s, d))

    # ---- SOFT (dressing) jobs: SMALL size, negative wear delta ----
    # These are the disguised maintenance actions. Because their SIZE is small,
    # any size-priority heuristic buries them at the end where their restorative
    # value is wasted.  Their |delta| capacity is deliberately LESS than the
    # total damage of the hard jobs, so wear cannot be fully cancelled by soft
    # jobs alone -- a strong schedule must ALSO spend a few re-dresses.
    for _ in range(n_soft):
        s = rng.randint(1, 3)
        d = -rng.choice([1, 2])
        jobs.append((s, d))

    # sanity: total damage must exceed soft restorative capacity so that resets
    # remain part of any good schedule (keeps the reset-cost tradeoff live).
    tot_pos = sum(d for (_, d) in jobs if d > 0)
    tot_neg = -sum(d for (_, d) in jobs if d < 0)
    # (with the ranges above this holds comfortably; assert to be safe)
    assert tot_pos > tot_neg, (tot_pos, tot_neg)

    # ---- ADVERSARIAL input order (this is the checker's baseline order) ----
    # Put the hard jobs first, sorted by ASCENDING size so the biggest hard jobs
    # run when wear is already highest; append all soft jobs afterwards where
    # their maintenance value is completely wasted. Processing in this given
    # order with no re-dress is a bad-but-feasible schedule -> the baseline.
    hard = sorted([j for j in jobs if j[1] > 0], key=lambda x: x[0])
    soft = sorted([j for j in jobs if j[1] < 0], key=lambda x: x[0])
    order = hard + soft

    lines = ["%d %d %d" % (N, T_r, W_max)]
    for (s, d) in order:
        lines.append("%d %d" % (s, d))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    gen(int(sys.argv[1]))

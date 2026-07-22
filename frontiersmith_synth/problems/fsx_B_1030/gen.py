import sys, random

# ---------------------------------------------------------------------------
# cherrypick-rate-board (format C, maximize net harvest = value picked minus
# wages paid).  `python3 gen.py <testId>` prints ONE instance to stdout.
# Deterministic in testId only.
#
# Instance:
#   line 1:  N M C_UNIT RMAX
#   next N lines:  "v d t"     field value, difficulty (1..10), deadline (1..M)
#   next M lines:  "s w"       worker skill (1..20), reservation wage -- given
#                              IN ARRIVAL ORDER (line k = the k-th worker to
#                              arrive)
#
# cost(field i, worker j) = ceil(C_UNIT * d_i / s_j)   (comparative advantage:
# higher skill -> cheaper on hard fields)
# ---------------------------------------------------------------------------

C_UNIT = 20
RMAX = 300
DMAX = 10
SKMAX = 20

# ladder sizes: (N fields, M workers)
SIZES = {
    1: (30, 70),   2: (45, 100),  3: (60, 140),  4: (70, 160),  5: (90, 200),
    6: (100, 220), 7: (130, 280), 8: (150, 320), 9: (180, 380), 10: (200, 420),
}

# rungs that plant the arrival-order / adverse-sorting trap: a handful of
# hard, valuable fields that only LATE-arriving elite workers can do cheaply,
# while a wave of low-skill workers arrives FIRST.
TRAP_IDS = {3, 5, 7, 9}


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    rng = random.Random(100003 + 97 * t)

    N, M = SIZES[t]
    trap = t in TRAP_IDS

    n_hard = max(3, N // 5) if trap else 0
    elite_start = int(M * 0.65)

    fields = []  # [v, d, deadline]
    for i in range(N):
        if trap and i < n_hard:
            v = rng.randint(300, 420)
            d = rng.randint(8, DMAX)
            if i % 2 == 0:
                # generous deadline: still reachable once the elites arrive
                dl = rng.randint(min(M, elite_start + 5), M)
            else:
                # tight deadline: must be handled BEFORE any elite arrives ->
                # forces a deliberate, targeted overpay-to-intercept decision
                lo = max(1, elite_start - M // 6)
                hi = max(lo, elite_start - 1)
                dl = rng.randint(lo, hi)
        else:
            v = rng.randint(80, 280)
            d = rng.randint(1, 7)
            dl = rng.randint(max(1, M // 4), M)
        fields.append([v, d, dl])
    rng.shuffle(fields)

    workers = []  # [skill, wage] in arrival order
    if trap:
        n_elite = n_hard
        elite_positions = set()
        late_len = M - elite_start
        step = max(1, late_len // max(1, n_elite))
        pos = elite_start
        for _ in range(n_elite):
            elite_positions.add(min(M - 1, pos))
            pos += step
        for j in range(M):
            wage = rng.randint(0, 10)
            if j in elite_positions:
                skill = rng.randint(18, SKMAX)
            elif j < elite_start:
                skill = rng.randint(1, 6)
            else:
                skill = rng.randint(9, 17)
            workers.append([skill, wage])
    else:
        for j in range(M):
            skill = rng.randint(1, SKMAX)
            wage = rng.randint(0, 15)
            workers.append([skill, wage])

    out = []
    out.append("%d %d %d %d" % (N, M, C_UNIT, RMAX))
    for (v, d, dl) in fields:
        out.append("%d %d %d" % (v, d, dl))
    for (s, w) in workers:
        out.append("%d %d" % (s, w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

import sys, random

# ---- ballistic-deposition-relief instance generator -----------------------
# A 1D substrate of L columns grows under aimed flux over T time steps. Each
# shot is "aimed" at a column, but a SHADOWING rule can redirect it: within a
# capture radius R around the aim, a column catches the particle away from
# the aim only if it is MORE THAN MARGIN M taller than the aim column itself
# (ties among qualifying stealers -> nearest to the aim, then leftmost);
# otherwise the aim just catches its own particle. The particle then sticks
# by the standard ballistic-deposition lateral rule at whichever column
# actually caught it. Budget T is deliberately SHORT of sum(target) (a
# fixed fraction of it) -- there is never enough flux to hit every column's
# target exactly, so which columns to under-serve is a genuine, no-free-
# lunch allocation choice on top of the shadowing dynamics: a perfectly
# shadow-aware scheduler still cannot reach every target, only decide where
# the unavoidable shortfall lands.
#
# The margin M means ordinary, mildly-varying relief (adjacent columns a
# few units apart) grows exactly as aimed -- "deposit to shape" works fine
# most of the time. TRAP testIds plant an isolated column MANY units taller
# than M above its neighbors (within R). Once that column clears the margin,
# every shot aimed at its low neighbor gets captured by it instead (it is
# the qualifying max), which only makes the tall column taller and the gap
# grow forever -- an obvious "aim at the biggest deficit" recipe walks
# straight into this regime change without noticing and dumps budget into
# runaway overshoot while the neighboring valley never moves. A solver that
# SIMULATES where a shot actually lands before choosing it (the
# shadowing-aware, anticipatory strategy) recognizes the trap and steers
# budget to columns it can still genuinely influence instead of feeding the
# runaway peak.

MARGIN = 3


def smooth_profile(L, maxamp, rng):
    v = rng.randint(0, max(1, maxamp // 3))
    prof = []
    for _ in range(L):
        v += rng.choice([-1, 0, 0, 1])
        v = max(0, min(maxamp, v))
        prof.append(v)
    return prof


def peaks_profile(L, R, rng, npeaks, peak_h, valley_lo, valley_hi):
    prof = [rng.randint(valley_lo, valley_hi) for _ in range(L)]
    gap = 2 * R + 3
    if npeaks * gap > L:
        npeaks = max(1, L // gap)
    positions = []
    cursor = rng.randint(0, gap - 1)
    for _ in range(npeaks):
        if cursor >= L:
            break
        positions.append(cursor)
        cursor += gap
    for p in positions:
        prof[p] = peak_h
        # keep the immediate shoulders (within R) genuinely low so the
        # peak-vs-valley adjacency inside the capture window is real
        for off in range(1, R + 1):
            if p - off >= 0:
                prof[p - off] = min(prof[p - off], valley_hi)
            if p + off < L:
                prof[p + off] = min(prof[p + off], valley_hi)
    return prof


def staircase_profile(L, rng, lo, hi):
    prof = []
    cur_hi = True
    for _ in range(L):
        if cur_hi:
            prof.append(rng.randint((hi + lo) // 2, hi))
        else:
            prof.append(rng.randint(lo, max(lo, (hi + lo) // 2 - 1)))
        cur_hi = not cur_hi
    return prof


def comb_profile(L, hi, lo):
    return [hi if i % 2 == 0 else lo for i in range(L)]


BUDGET_FRAC = {
    1: 0.85, 2: 0.85, 3: 0.7, 4: 0.65, 5: 0.65,
    6: 0.7, 7: 0.6, 8: 0.65, 9: 0.7, 10: 0.6,
}


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260718 + 97 * t)

    if t == 1:
        L, R = 12, 1
        target = smooth_profile(L, 4, rng)
    elif t == 2:
        L, R = 16, 1
        target = smooth_profile(L, 5, rng)
    elif t == 3:
        L, R = 18, 2
        target = smooth_profile(L, 6, rng)
        # sprinkle in one mild bump
        p = rng.randint(2, L - 3)
        target[p] = min(9, target[p] + 4)
    elif t == 4:
        # TRAP: isolated tall peak next to a low but NOT-trivial valley --
        # the valley needs several units of height, so a recipe that insists
        # on finishing one column's whole quota before moving on repeats the
        # capture mistake valley_target times in a row. L is generous
        # relative to the trapped zone so there is always plenty of
        # legitimately-reachable substrate elsewhere to route unavoidable
        # excess budget into (a competent solver should still clear the
        # baseline comfortably; only the naive recipe should not).
        L, R = 26, 2
        target = peaks_profile(L, R, rng, npeaks=2, peak_h=12, valley_lo=2, valley_hi=3)
    elif t == 5:
        # TRAP: several isolated peaks, each shadowing its own multi-unit valley
        L, R = 30, 2
        target = peaks_profile(L, R, rng, npeaks=3, peak_h=11, valley_lo=2, valley_hi=3)
    elif t == 6:
        L, R = 18, 1
        target = smooth_profile(L, 6, rng)
        p = rng.randint(2, L - 3)
        target[p] = min(9, target[p] + 3)
    elif t == 7:
        # TRAP: big-radius staircase -- large jumps between adjacent columns
        L, R = 24, 3
        target = staircase_profile(L, rng, lo=0, hi=11)
    elif t == 8:
        # moderate: alternating comb -- lots of local height variation,
        # but every column is close to at least one same-height neighbor,
        # so this is easier than the isolated-peak traps above.
        L, R = 22, 3
        target = comb_profile(L, hi=9, lo=1)
    elif t == 9:
        L, R = 30, 2
        target = smooth_profile(L, 6, rng)
        for _ in range(2):
            p = rng.randint(2, L - 3)
            target[p] = min(10, target[p] + 4)
    else:
        # TRAP: largest, hardest -- mixed peaks + staircase segments
        L, R = 44, 3
        half = L * 5 // 8
        a = peaks_profile(half, R, rng, npeaks=3, peak_h=11, valley_lo=2, valley_hi=3)
        b = staircase_profile(L - half, rng, lo=0, hi=9)
        target = a + b

    target = [max(0, min(15, x)) for x in target]
    total = sum(target)
    T = max(1, round(BUDGET_FRAC[t] * total))

    out = [f"{L} {R} {MARGIN} {T}", " ".join(map(str, target))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""gen.py <testId> -- prints one cleanroom-airlock-caste instance to stdout.
Deterministic: all randomness seeded from testId only.

Instance format:
  L R J T C KCOST
  J lines, one per job (job id = 1-based line index):
    grade release duration deadline weight
"""
import random
import sys


def emit(testId):
    rng = random.Random(20240 + 7 * testId)

    L = 4          # grade levels, 1 (dirtiest) .. L (cleanest)
    T = 6          # decon airlock cycle duration (ticks)
    C = 3          # decon airlock capacity (robots per cycle)
    KCOST = 120    # cost charged per decon cycle used

    R = 3 + testId                # 4 .. 13
    J = 10 * testId + 6           # 16 .. 106 (always enough waves to hit a surge)

    wave = max(2, R // 3)        # normal wave size (robots freeing up together)

    lines = [f"{L} {R} {J} {T} {C} {KCOST}"]

    # Steady-state grade mix: a fixed skew so a workload-proportional caste
    # partition is meaningful (not perfectly uniform, not degenerate).
    skew = [3, 2, 2, 1]           # relative frequency weight, grade 1..L
    grade_cycle = []
    for g in range(1, L + 1):
        grade_cycle += [g] * skew[g - 1]
    rng.shuffle(grade_cycle)      # fixed (seeded) shuffle -> deterministic
    gi = 0
    burst_weights = [3, 2, 2, 1]  # which grade a surge lands on (weighted, seeded)

    jid = 0
    wave_start = 0
    wave_no = 0
    while jid < J:
        wave_no += 1
        is_surge = (wave_no % 3 == 0)
        if is_surge:
            # A surge: many jobs of the SAME grade land almost simultaneously
            # -- more than any workload-proportional static caste for that
            # grade can absorb without a queue, and the queue eats the tight
            # slack. No static partition escapes this without either some
            # lateness or a floater assist (which itself costs a T-tick
            # decon >> the slack budget). This is what stops ANY strategy,
            # including the intended strong one, from reaching zero cost.
            burst_g = rng.choices(range(1, L + 1), weights=burst_weights, k=1)[0]
            size = wave * 3
        else:
            size = wave

        for _ in range(size):
            if jid >= J:
                break
            jid += 1
            if is_surge:
                g = burst_g
                release = wave_start + rng.randint(0, 1)
                slack = rng.randint(2, 3)
            else:
                g = grade_cycle[gi % len(grade_cycle)]
                gi += 1
                release = wave_start + rng.randint(0, 2)
                slack = rng.randint(2, 5)
            dur = rng.randint(4, 9)
            deadline = release + dur + slack
            weight = rng.choice([1, 2, 2, 3, 5])
            lines.append(f"{g} {release} {dur} {deadline} {weight}")
        wave_start += rng.randint(2, 4)

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1])
    emit(tid)

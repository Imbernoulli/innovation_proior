import sys, math, random

# Difficulty ladder + planted traps, all keyed off testId only (deterministic).
TRAP_LATENCY_IDS = {4, 6, 8, 10}   # cases where variant 0 gets long signal latency + tight window
CAP_LO, CAP_HI = 5.0, 6.0          # fleet size = (5..6)x the one-shot safe cohort size


def main():
    i = int(sys.argv[1])
    rng = random.Random(9000 + i)

    if i <= 2:
        V, T = 2, 12
    elif i <= 5:
        V, T = 3, 16
    elif i <= 8:
        V, T = 3, 20
    else:
        V, T = 4, 24

    lines = [f"{V} {T}"]
    for v in range(V):
        reward = round(rng.uniform(0.9, 1.1), 3)
        is_trap_latency = (v == 0 and i in TRAP_LATENCY_IDS)
        is_bad_ev = (v == V - 1 and i % 3 == 0)  # a planted negative-expected-value variant

        if is_bad_ev:
            p = round(rng.uniform(0.12, 0.22), 4)
            target_ev = round(rng.uniform(-0.6, -0.2), 3)
        else:
            p = round(rng.uniform(0.03, 0.10), 4)
            target_ev = round(rng.uniform(0.3, 0.7), 3)

        # Solve penalty so that (1-p)*reward - p*penalty == target_ev exactly.
        penalty = ((1 - p) * reward - target_ev) / p
        penalty = round(max(1.0, penalty), 3)

        if is_trap_latency:
            D = rng.randint(6, 9)
            W = round(rng.uniform(1.5, 3.0), 3)
        else:
            D = rng.randint(1, 3)
            W = round(rng.uniform(4.0, 12.0), 3)

        singleshot = math.floor(W / p)
        mult = rng.uniform(CAP_LO, CAP_HI)
        fleet = max(singleshot + 10, int(singleshot * mult))

        lines.append(f"{fleet} {p:.6f} {reward:.6f} {penalty:.6f} {D} {W:.6f}")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

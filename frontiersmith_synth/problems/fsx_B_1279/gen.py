import sys, random

# Deterministic difficulty/trap ladder.  testId in {2,4,5,7,9,10} plants a single sharp
# legacy-liability peak that clusters rollover risk when the maturity ladder mirrors it
# exactly (the "cashflow matching" trap); the remaining testIds are flatter control cases.
TRAP_SET = {2, 4, 5, 7, 9, 10}


def main():
    i = int(sys.argv[1])
    rng = random.Random(20260000 + i)

    T = 10 + i                       # 11..20 periods
    L = [rng.randint(8, 12) for _ in range(T)]
    base_sum = sum(L)

    is_trap = i in TRAP_SET
    haircut_num, haircut_den = rng.choice([(1, 3), (2, 5), (1, 2)])

    L0mean = base_sum / T
    K = 2.5  # headroom factor: normal-condition capacity vs. baseline per-period liability
    Base = [max(6, round(L0mean * K * haircut_den / haircut_num) + rng.randint(-1, 1))
            for _ in range(T)]

    if is_trap:
        peak_t = rng.randint(2, T - 4)
        bump = rng.randint(int(0.5 * base_sum), int(0.8 * base_sum))
        L[peak_t] += bump

    F = sum(L)
    y = [rng.randint(300, 700) for _ in range(T)]   # yield, hundredths of a percent
    window_len = max(2, T // 5)
    S = 8                                            # hidden stress scenarios (see checker)

    lines = [
        f"{T} {F}",
        " ".join(map(str, L)),
        " ".join(map(str, y)),
        " ".join(map(str, Base)),
        f"{haircut_num} {haircut_den}",
        f"{S} {window_len}",
    ]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

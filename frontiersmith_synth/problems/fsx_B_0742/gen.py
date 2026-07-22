import sys, random

# Difficulty/trap ladder: (N units, T time-steps)
LADDER = [
    (4, 6), (4, 8), (5, 10), (5, 12), (6, 14),
    (6, 16), (7, 18), (7, 20), (8, 24), (9, 28),
]
# Test ids engineered so the N-1 spinning-reserve rule binds hard on the
# reserve-eligible ("fast") group AND that group mixes one very flat
# (cheap-to-displace) unit with several sharply-peaked (expensive-to-
# displace) ones that are ALSO the cheapest by linear fuel rate -- this is
# exactly the setup that rewards curvature-aware reserve allocation and
# punishes both plain merit-order dispatch and curvature-blind repair.
TRAP_IDS = {4, 5, 7, 9, 10}


def gen_instance(test_id):
    rng = random.Random(1000003 * test_id + 7)
    N, T = LADDER[test_id - 1]

    caps = [rng.randint(80, 260) for _ in range(N)]
    J = max(range(N), key=lambda i: caps[i])  # the swing/largest unit

    is_trap = test_id in TRAP_IDS

    ms, as_, bs, fast = [0.0] * N, [0.0] * N, [0.0] * N, [0] * N
    for i in range(N):
        ms[i] = round(rng.uniform(0.40, 0.65) * caps[i], 4)

    # pick the "fast" (reserve-eligible) group: all non-J units except a
    # small slow subset that cannot contribute to spinning reserve
    others = [i for i in range(N) if i != J]
    rng.shuffle(others)
    n_slow = 1 if N <= 5 else rng.randint(1, max(1, N // 3))
    slow_set = set(others[:n_slow])
    for i in range(N):
        if i == J or i in slow_set:
            fast[i] = 0
        else:
            fast[i] = 1
    if sum(fast) == 0:
        fast[others[0]] = 1

    fast_others = [i for i in range(N) if fast[i] == 1]
    flex_unit = rng.choice(fast_others) if fast_others else None

    for i in range(N):
        if is_trap:
            if i == flex_unit or i == J:
                a = rng.uniform(0.0005, 0.0012)   # nearly flat: cheap to displace
            elif fast[i] == 1:
                a = rng.uniform(0.25, 0.55)        # sharply peaked: expensive to displace
            else:
                a = rng.uniform(0.003, 0.012)
            if fast[i] == 1:
                b = rng.uniform(0.4, 0.9)          # fast units are also the cheapest
            else:
                b = rng.uniform(2.2, 3.5)
        else:
            a = rng.uniform(0.002, 0.02)
            b = rng.uniform(0.8, 3.0)
        as_[i] = round(a, 6)
        bs[i] = round(b, 4)

    cap_total = sum(caps)
    max_d = cap_total - caps[J]

    ds = []
    for _t in range(T):
        if is_trap:
            frac = rng.uniform(0.78, 0.93)
        else:
            frac = rng.uniform(0.35, 0.75)
        ds.append(round(frac * max_d, 4))

    return N, T, caps, ms, as_, bs, fast, J, ds


def main():
    test_id = int(sys.argv[1])
    N, T, caps, ms, as_, bs, fast, J, ds = gen_instance(test_id)
    out = [f"{N} {T}"]
    for i in range(N):
        out.append(f"{caps[i]} {ms[i]} {as_[i]} {bs[i]} {fast[i]}")
    out.append(str(J + 1))
    out.append(" ".join(f"{d}" for d in ds))
    print("\n".join(out))


if __name__ == "__main__":
    main()

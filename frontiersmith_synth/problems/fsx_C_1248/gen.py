import sys, random

# Pipeline stage-partition instance generator. testId 1..10 = difficulty ladder,
# small -> large/adversarial. testIds in TRAP_IDS are engineered so that the
# "maximize stage count" (peak clock frequency) strategy is FAR from optimal:
# resolve_block / hazard blocks are pushed to the far end of the datapath and
# the forwarding Budget is deliberately tight, so deep partitions blow their
# branch-flush and hazard-stall costs while a moderate partition (found by a
# joint stage-count/forwarding search) wins comfortably.

SIZES = {1: 6, 2: 7, 3: 8, 4: 8, 5: 9, 6: 9, 7: 10, 8: 10, 9: 12, 10: 13}
TRAP_IDS = {3, 5, 6, 8, 9, 10}


def k_for(n):
    if n <= 8:
        return 3
    if n <= 10:
        return 4
    return 5


def build(test_id):
    rng = random.Random(90210 + 7 * test_id)
    N = SIZES[test_id]
    K = k_for(N)
    trap = test_id in TRAP_IDS

    c = [rng.randint(1, 6) for _ in range(N)]
    L = rng.randint(1, 2)
    I = rng.randint(9 * N, 13 * N)

    if trap:
        resolve_block = N
        Br = rng.randint(max(1, I // 4), max(2, I // 2))
        Mb = Br  # every branch mispredicts
    else:
        resolve_block = rng.randint(1, max(1, N // 3))
        Br = rng.randint(max(1, I // 12), max(2, I // 8))
        Mb = rng.randint(1, max(1, Br // 8))

    haz = []
    forced = min(K - 1, 3) if trap else 0
    for k in range(K):
        if k < forced:
            need_b, res_b, dist = 1, N, 0
            freq = rng.randint(max(1, I // 4), max(2, I // 2))
        elif trap:
            need_b = rng.randint(1, max(1, N // 2))
            res_b = rng.randint(need_b + 1, N)
            dist = rng.randint(0, 2)
            freq = rng.randint(1, max(2, I // 14))
        else:
            # non-trap: real but modest stalls -- a deep partition is still
            # decent, but the forwarding subset choice is genuinely binding.
            need_b = rng.randint(1, max(1, N // 2))
            res_b = rng.randint(need_b + 1, N)
            dist = rng.randint(0, max(0, N // 2))
            freq = rng.randint(max(1, I // 20), max(2, I // 10))
        haz.append((need_b, res_b, dist, freq))

    if trap:
        Budget = rng.randint(max(1, N // 5), max(2, N // 3))
    else:
        Budget = rng.randint(max(2, (4 * N) // 5), N)

    lines = []
    lines.append(f"{N} {K} {L} {Br} {Mb} {resolve_block} {Budget} {I}")
    lines.append(" ".join(map(str, c)))
    for (need_b, res_b, dist, freq) in haz:
        lines.append(f"{need_b} {res_b} {dist} {freq}")
    return "\n".join(lines) + "\n"


def main():
    test_id = int(sys.argv[1])
    sys.stdout.write(build(test_id))


if __name__ == "__main__":
    main()

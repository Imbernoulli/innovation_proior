import random
import sys


# Difficulty ladder: (N, H, cluster_lo, cluster_hi, extra_bg, M)
#  N         = number of prep steps (chain length)
#  H         = number of planted "hot zone" bursts of expensive steps
#  cluster_* = number of serve-requests clustered right after each hot zone
#  extra_bg  = number of extra background serve-requests scattered anywhere
#  M         = live-burner cap, DELIBERATELY undersupplied relative to H so
#              not every hot zone can be checkpointed -- even the smart
#              solver must choose which zones earn a burner.
SIZES = [
    (40, 3, 2, 3, 3, 4),
    (80, 4, 2, 3, 4, 4),
    (150, 5, 3, 4, 5, 5),
    (300, 6, 3, 4, 6, 5),
    (500, 8, 3, 5, 8, 6),
    (800, 10, 4, 5, 10, 6),
    (1200, 12, 4, 5, 12, 7),
    (1800, 14, 4, 6, 14, 6),
    (2400, 16, 4, 6, 16, 6),
    (3000, 18, 5, 6, 18, 6),
]


def build(testId):
    N, H, clo, chi, extra_bg, M = SIZES[(testId - 1) % len(SIZES)]
    rnd = random.Random(900000 + 7919 * testId)

    costs = [0] * (N + 1)  # 1-indexed
    for i in range(1, N + 1):
        costs[i] = rnd.randint(2, 6)

    # Place H irregular hot zones, each followed by a cheap "regenerator" node.
    lo_bound = 4
    hi_bound = max(lo_bound + H + 2, int(N * 0.82))
    starts = sorted(rnd.sample(range(lo_bound, hi_bound), H))
    zones = []
    prev_limit = 0
    for idx, s in enumerate(starts):
        s = max(s, prev_limit + 3)
        max_len = max(4, N // 55)
        length = rnd.randint(3, max_len)
        next_cap = starts[idx + 1] - 6 if idx + 1 < len(starts) else N - 4
        e = min(s + length, max(s, next_cap))
        if e <= s:
            e = s
        for j in range(s, e + 1):
            if j <= N:
                costs[j] = rnd.randint(150, 900)
        reg = min(e + 1, N)
        costs[reg] = 1
        zones.append((s, e, reg))
        prev_limit = reg + 2

    requests = set()
    for (s, e, reg) in zones:
        window_hi = min(N, reg + max(6, N // 40))
        window_lo = min(N, reg + 1)
        if window_lo > window_hi:
            continue
        k = rnd.randint(clo, chi)
        span = list(range(window_lo, window_hi + 1))
        rnd.shuffle(span)
        for p in span[:k]:
            requests.add(p)

    tries = 0
    while len(requests) < extra_bg + H and tries < 2000:
        p = rnd.randint(1, N)
        requests.add(p)
        tries += 1

    req_list = sorted(requests, reverse=True)
    K = len(req_list)

    out = []
    out.append(f"{N} {M} {K}")
    out.append(" ".join(str(costs[i]) for i in range(1, N + 1)))
    out.append(" ".join(str(x) for x in req_list))
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(build(testId))


if __name__ == "__main__":
    main()

import sys, random


def build_instance(testId):
    rng = random.Random(20000 + testId * 97)
    Ms = {1: 6, 2: 6, 3: 8, 4: 8, 5: 10, 6: 10, 7: 12, 8: 12, 9: 16, 10: 16}
    M = Ms[testId]
    K = 5
    costs = []
    markups = []
    tiers = []
    for j in range(M):
        tier = j % 4
        cost = 5 + 10 * tier + rng.randint(0, 4)
        markup = 10 + 12 * tier + rng.randint(0, 6)
        costs.append(cost)
        markups.append(markup)
        tiers.append(tier)

    pop_scale = 2 + (testId - 1) // 2
    base_pop = [3, 6, 10, 6, 3]
    cohorts = []  # list (per cohort) of list of (n, o, v[])
    for k in range(K):
        wealth = k
        pref_main = wealth
        pref_sec = max(wealth - 1, 0)
        total_pop = base_pop[k] * pop_scale + rng.randint(0, 2)
        n_main = max(1, int(round(total_pop * 0.65)))
        n_sec = max(1, total_pop - n_main)
        types = []
        for (pref, n) in [(pref_main, n_main), (pref_sec, n_sec)]:
            o = 0
            if k == 0:
                o = 2
            if k == K - 1:
                o = 14 + testId
            v = []
            for j in range(M):
                tier = tiers[j]
                base = 25 + 18 * tier
                wb = 10 * wealth
                mismatch = 13 * abs(tier - pref)
                jit = rng.randint(-3, 3)
                val = max(1, base + wb - mismatch + jit)
                v.append(val)
            types.append((n, o, v))
        cohorts.append(types)
    return M, K, costs, markups, cohorts


def main():
    testId = int(sys.argv[1])
    M, K, costs, markups, cohorts = build_instance(testId)
    out = []
    out.append(f"{M} {K}")
    for j in range(M):
        out.append(f"{costs[j]} {markups[j]}")
    for k in range(K):
        types = cohorts[k]
        out.append(f"{len(types)}")
        for (n, o, v) in types:
            out.append(f"{n} {o} " + " ".join(str(x) for x in v))
    print("\n".join(out))


if __name__ == "__main__":
    main()

import sys, random, itertools


def in_subcube(s):
    # baseline "shoreline ridge" plan: first digit 0, all other digits in {0,1}
    return s[0] == '0' and all(c in '01' for c in s[1:])


def main():
    t = int(sys.argv[1])
    ladder = {1: 3, 2: 3, 3: 4, 4: 4, 5: 5, 6: 5, 7: 6, 8: 6, 9: 7, 10: 7}
    n = ladder.get(t, 3 + ((t - 1) % 5))
    seed = t
    frac = 0.10 + 0.05 * ((t - 1) % 3)   # 0.10 / 0.15 / 0.20 cycling

    strs = [''.join(map(str, v)) for v in itertools.product(range(3), repeat=n)]
    sub = set(s for s in strs if in_subcube(s))
    pool = [s for s in strs if s not in sub]      # never block the baseline ridge

    rng = random.Random(1000 + seed)
    b = min(int(frac * len(strs)), len(pool))
    blocked = sorted(rng.sample(pool, b))

    wr = random.Random(5000 + seed)
    weights = [wr.randint(1, 9) for _ in strs]

    out = [str(n), str(len(blocked))]
    out.extend(blocked)
    out.append(' '.join(map(str, weights)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()

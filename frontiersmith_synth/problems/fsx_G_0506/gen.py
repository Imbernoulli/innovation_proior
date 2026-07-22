#!/usr/bin/env python3
"""Generate one deterministic sparse-polynomial factorization instance.

The printed target polynomials are expanded, but are assembled from planted
blocks of the form core_b * (signed sparse linear form).  TestId alone controls
the instance.
"""
import random
import sys


def params(test_id):
    n = 9 + 2 * test_id              # 11 .. 29
    m = 3 + (test_id + 1) // 2       # 4 .. 8
    blocks = 2 + test_id // 2        # 2 .. 7
    tails = 4 + test_id // 3         # 4 .. 7
    core_deg = 3 + (test_id % 3)     # 3 .. 5
    noise = 1 + test_id // 4         # 1 .. 3
    return n, m, blocks, tails, core_deg, noise


def choose_blocks(rng, n, blocks, tails, core_deg):
    data = []
    seen_cores = set()
    seen_full = set()
    attempts = 0
    while len(data) < blocks and attempts < 10000:
        attempts += 1
        core = tuple(sorted(rng.sample(range(n), core_deg)))
        if core in seen_cores:
            continue
        candidates = [v for v in range(n) if v not in core]
        if len(candidates) < tails:
            continue
        tail_vars = tuple(sorted(rng.sample(candidates, tails)))
        fulls = {tuple(sorted(core + (v,))) for v in tail_vars}
        if fulls & seen_full:
            continue
        seen_cores.add(core)
        seen_full.update(fulls)
        data.append((core, tail_vars))
    if len(data) != blocks:
        raise RuntimeError("could not build disjoint planted blocks")
    return data, seen_full


def random_noise(rng, n, degree, forbidden):
    for _ in range(10000):
        d = rng.randint(2, min(degree, n))
        mono = tuple(sorted(rng.sample(range(n), d)))
        if mono not in forbidden:
            forbidden.add(mono)
            return mono
    raise RuntimeError("could not draw unique noise monomial")


def build(test_id):
    rng = random.Random(506000 + 7919 * test_id)
    n, m, blocks_n, tails_n, core_deg, noise_n = params(test_id)
    blocks, forbidden = choose_blocks(rng, n, blocks_n, tails_n, core_deg)
    targets = []

    for j in range(m):
        use_count = min(blocks_n, 2 + ((j + test_id) % 3) + (1 if test_id >= 7 else 0))
        start = (2 * j + test_id) % blocks_n
        use = [(start + r) % blocks_n for r in range(use_count)]
        terms = []
        local = set()

        for bpos, bi in enumerate(use):
            core, tail_vars = blocks[bi]
            for ti, tail in enumerate(tail_vars):
                mono = tuple(sorted(core + (tail,)))
                if mono in local:
                    continue
                local.add(mono)
                # The first term in every target is positive, keeping the
                # reference baseline exactly aligned with trivial.py.
                if not terms:
                    coeff = 1
                else:
                    coeff = -1 if ((17 * j + 5 * bi + 3 * ti + test_id + bpos) % 4 == 0) else 1
                terms.append((coeff, mono))

        for z in range(noise_n):
            mono = random_noise(rng, n, core_deg + 1, forbidden)
            if mono in local:
                continue
            local.add(mono)
            coeff = -1 if ((j + z + test_id) % 3 == 0) else 1
            terms.append((coeff, mono))

        targets.append(terms)

    return n, m, targets


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n, m, targets = build(test_id)
    out = ["%d %d" % (n, m)]
    for terms in targets:
        out.append(str(len(terms)))
        for coeff, mono in terms:
            out.append("%d %d %s" % (coeff, len(mono), " ".join(str(v) for v in mono)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

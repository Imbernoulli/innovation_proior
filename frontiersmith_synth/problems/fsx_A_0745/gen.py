import sys, random

ALPHA = "01234567"


def rand_atom(rng, lo=3, hi=6):
    L = rng.randint(lo, hi)
    return "".join(rng.choice(ALPHA) for _ in range(L))


def mutate_str(rng, s):
    if not s:
        return s
    pos = rng.randrange(len(s))
    old = s[pos]
    choices = [c for c in ALPHA if c != old]
    newc = rng.choice(choices)
    return s[:pos] + newc + s[pos + 1:]


def build(rng, n_atoms, n_blocks, n_sections, block_len_range, section_len_range,
          target_len, mut_rate, noisy_prob):
    atoms = [rand_atom(rng) for _ in range(n_atoms)]
    blocks_pat = [[rng.randrange(n_atoms) for _ in range(rng.randint(*block_len_range))]
                  for _ in range(n_blocks)]

    def block_instance(bi, noisy):
        idxs = blocks_pat[bi]
        parts = []
        for ai in idxs:
            a = atoms[ai]
            if noisy and rng.random() < mut_rate:
                a = mutate_str(rng, a)
            parts.append(a)
        return "".join(parts)

    sections_pat = [[rng.randrange(n_blocks) for _ in range(rng.randint(*section_len_range))]
                    for _ in range(n_sections)]

    def section_instance(si, noisy):
        return "".join(block_instance(bi, noisy) for bi in sections_pat[si])

    pieces = []
    total = 0
    idx = 0
    # cycle through sections (occasionally noisy) until we reach target length
    while total < target_len:
        si = idx % n_sections
        noisy = rng.random() < noisy_prob
        piece = section_instance(si, noisy)
        pieces.append(piece)
        total += len(piece)
        idx += 1

    s = "".join(pieces)
    if len(s) > target_len:
        s = s[:target_len]
    elif len(s) < target_len:
        s += "".join(rng.choice(ALPHA) for _ in range(target_len - len(s)))
    return s


# difficulty ladder: (n_atoms, n_blocks, n_sections, block_len_range, section_len_range,
#                      target_len, mut_rate, noisy_prob)
LADDER = [
    (3, 2, 2, (3, 5), (2, 3), 40, 0.05, 0.10),
    (4, 3, 2, (4, 6), (2, 4), 100, 0.08, 0.15),
    (5, 4, 3, (4, 7), (3, 4), 250, 0.10, 0.20),
    (6, 5, 3, (5, 8), (3, 5), 600, 0.12, 0.25),
    (7, 6, 4, (5, 8), (4, 6), 1200, 0.15, 0.30),
    (8, 7, 4, (6, 9), (4, 6), 2500, 0.18, 0.35),
    (9, 8, 5, (6, 9), (5, 7), 4500, 0.22, 0.40),
    (10, 9, 5, (6, 9), (5, 7), 7000, 0.28, 0.45),
    (12, 10, 6, (6, 9), (5, 8), 9500, 0.32, 0.50),
    (14, 12, 6, (6, 9), (5, 8), 12000, 0.35, 0.55),
]


def main():
    i = int(sys.argv[1])
    i = max(1, min(len(LADDER), i))
    (n_atoms, n_blocks, n_sections, blr, slr,
     target_len, mut_rate, noisy_prob) = LADDER[i - 1]
    rng = random.Random(914200 + i * 97)
    s = build(rng, n_atoms, n_blocks, n_sections, blr, slr, target_len, mut_rate, noisy_prob)
    n = len(s)
    sys.stdout.write(str(n) + "\n" + s + "\n")


if __name__ == "__main__":
    main()

import sys, random

BASES = "ACGT"


def main():
    testId = int(sys.argv[1])
    rng = random.Random(20260000 + testId * 97)

    P = 20           # probe length (fixed schema)
    W = 50           # marker-locus window length (fixed schema)
    D = 2            # Hamming match threshold (fixed schema)

    M = 8 + testId                       # marker loci: 9..18
    C = max(2, M // 4)                   # never-edited "core" loci (also checker's baseline size)
    stress = M - C                       # loci that some published mutant may stress
    L = 4000 + testId * 1500             # reference length: 5500..19000

    spacing = (L - 400) // M
    min_spacing = W + P + 20
    if spacing < min_spacing:
        spacing = min_spacing
        L = 400 + spacing * M
    positions = [200 + i * spacing for i in range(M)]

    ref = "".join(rng.choice(BASES) for _ in range(L))

    # ---- SNP-cluster damage bands: shift 0 and shift (W-P) both hit >=3 damage offsets;
    #      exactly one shift range (near `low_end`) is damage-free (found only by a real search) ----
    low_end = rng.randint(10, 20)
    high_start = low_end + P
    damage_low = sorted(rng.sample(range(0, low_end), min(4, low_end)))
    hi_room = list(range(high_start, W))
    damage_high = sorted(rng.sample(hi_room, min(4, len(hi_room))))
    snp_offsets = sorted(damage_low + damage_high)

    # ---- tandem-retile parameters (same relative pattern reused at every affected locus) ----
    tlen = rng.randint(4, 7)
    tstart = rng.randint(0, W - tlen)

    # ---- published mutant panel ----
    stress_idx = list(range(C, M))
    mutants = []

    def add_class(cls, n_variants, frac_lo, frac_hi):
        lo = max(1, int(round(frac_lo * stress)))
        hi = max(lo, int(round(frac_hi * stress)))
        hi = min(hi, stress)
        for v in range(n_variants):
            s = hi if v == 0 else rng.randint(lo, hi)   # v==0: deterministic worst-case subset
            subset = rng.sample(stress_idx, s)
            mutants.append((cls, sorted(subset)))

    n_variants = 4 + testId // 2         # 4..9
    add_class(0, n_variants, 0.55, 0.75)  # SNP-cluster: large subsets
    add_class(1, n_variants, 0.55, 0.75)  # segment-inversion: large subsets
    add_class(2, n_variants, 0.15, 0.30)  # tandem-retile: small subsets

    K = C * 1 + stress * 2 + 6

    out = []
    out.append(f"{L} {M} {K} {P} {W} {D}")
    out.append(ref)
    out.append(" ".join(map(str, positions)))
    out.append(str(len(snp_offsets)))
    out.append(" ".join(map(str, snp_offsets)))
    out.append(f"{tlen} {tstart}")
    out.append(str(len(mutants)))
    for cls, subset in mutants:
        out.append(f"{cls} {len(subset)} " + " ".join(map(str, subset)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

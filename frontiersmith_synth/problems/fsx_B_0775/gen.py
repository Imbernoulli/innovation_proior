#!/usr/bin/env python3
"""gen.py <testId> -- deterministic generator for fsx_B_0775 (hash vs published key-family sweep).

Prints one instance:
    M F
    <family_1>
    ...
    <family_F>

Determinism: everything derives from testId only, via a fixed-seed LCG (no wall clock,
no OS randomness).
"""
import sys

MASK64 = (1 << 64) - 1
M = 1024

# The "textbook" Fibonacci-hashing constant that the greedy reference solution trusts
# blindly. It is baked into the generator ONLY so that a resonant AP-family (trap B) can
# be constructed against it; solvers never need to know this value to solve the problem.
A0 = 0x9E3779B97F4A7C15
A0_INV = pow(A0, -1, 1 << 64)

BASE_SEED = 4000


def lcg_stream(seed):
    x = seed & MASK64
    while True:
        x = (6364136223846793005 * x + 1442695040888963407) & MASK64
        yield x


def build_case(tid):
    rng = lcg_stream(BASE_SEED + tid)
    fams = []

    # --- generic dominant AP family (always present) ---
    start = next(rng)
    stride = next(rng) | 1  # odd stride -> full period mod any power of two
    count = 2000 + 150 * tid
    fams.append(("AP", start, stride, count))

    # --- generic bit-plane cluster (contiguous window, mostly low bits) ---
    base = next(rng)
    fams.append(("COSET", base, 2, 9))  # window bits [2,11)

    # --- generic low-entropy float-style cluster ---
    exp = next(rng) & ((1 << 24) - 1)
    mant_base = next(rng) & ((1 << 40) - 1)
    mant_stride = (next(rng) & ((1 << 40) - 1)) | 1
    fcount = 1200 + 80 * tid
    fams.append(("FLOAT", exp, mant_base, mant_stride, fcount))

    trapA_cases = {4, 8}
    trapB_cases = {2, 5, 7, 9}
    hardcoset_cases = {10}

    if tid in trapA_cases:
        # stride is a multiple of M: under a raw "mod M" reduction (no mixing stage
        # first), EVERY key in this family shares the same low 10 bits, so the whole
        # family collapses into a single bucket no matter which multiplier/offset a
        # MODM-only pipeline chooses. A pipeline that mixes bits before reducing (e.g.
        # a multiply + TOPBITS) is unaffected.
        mult = (next(rng) | 1) % 100000 + 1
        strideA = M * mult
        startA = next(rng)
        countA = 150 + 20 * tid
        fams.append(("AP", startA, strideA, countA))

    if tid in trapB_cases:
        # stride chosen so that A0 * stride == delta (mod 2^64) for a SMALL delta.
        # Any pipeline that multiplies by exactly A0 and reads the top 10 bits sees the
        # whole family crawl through only a few adjacent buckets (delta/2^54 is tiny),
        # regardless of how many keys there are. A pipeline that multiplies by a
        # DIFFERENT constant (chosen by looking at this specific sweep, not by trusting
        # one canonical constant) does not resonate.
        deltabits = 50
        delta = 1 << deltabits
        strideB = (A0_INV * delta) % (1 << 64)
        startB = next(rng)
        countB = 1200 + 60 * tid
        fams.append(("AP", startB, strideB, countB))

    if tid in hardcoset_cases:
        # large contiguous high-bit cluster: stresses raw MODM (all free bits >= 10)
        # and gives the biggest single family in the whole sweep.
        baseH = next(rng)
        fams.append(("COSET", baseH, 20, 15))

    return fams


def fmt_family(fam):
    kind = fam[0]
    if kind == "AP":
        _, start, stride, count = fam
        return f"AP {start} {stride} {count}"
    if kind == "COSET":
        _, base, lo, width = fam
        return f"COSET {base} {lo} {width}"
    if kind == "FLOAT":
        _, exp, mant_base, mant_stride, count = fam
        return f"FLOAT {exp} {mant_base} {mant_stride} {count}"
    raise ValueError("bad family kind")


def main():
    tid = int(sys.argv[1])
    fams = build_case(tid)
    out = [f"{M} {len(fams)}"]
    for fam in fams:
        out.append(fmt_family(fam))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

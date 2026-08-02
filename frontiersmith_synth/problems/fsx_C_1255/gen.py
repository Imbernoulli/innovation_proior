#!/usr/bin/env python3
"""gen.py <testId> -- prints one ADC resolution/rate/filter/gain allocation
instance to stdout.

Instance:
    NBINS BUDGET WFILT PFS NFLOOR FLO FHI
    f_1 p_1 ... f_NBINS p_NBINS

testId 1..10 is a difficulty ladder. 1-3 are narrowband warm-ups with a
generous conversion budget: the tones cluster near the target band, so
"maximize resolution B" still leaves the forced sample rate R high enough
that the antialias cutoff fc=R/2 comfortably covers the band -- the naive
recipe is not punished much. 4-10 (the trap majority, per the family spec)
are wideband: tones are scattered up to several times FHI and the budget is
deliberately tight, so B=16 forces R down far enough that fc=R/2 drops below
FHI. The antialias filter's cutoff is pinned to fc, so a starved R both (a)
attenuates the target band's own high tones and (b) still lets substantial
above-Nyquist interferer energy alias in -- "maximize resolution" aliases
the very band it was supposed to protect. All randomness is seeded from
testId only, so the same testId always reproduces byte-identical output.
"""
import sys
import random

# (nbins, flo, fhi, pfs, nfloor, wideband, fmax_mult, budget_mult, wfilt_div)
# NFLOOR is the fixed hardware noise floor added to every noise total, given
# directly as an integer (kept tiny for the narrowband warm-ups, where the
# quantization axis alone already separates greedy from the reference
# baseline; scaled up with PFS for the wideband traps so the achievable
# ceiling stays away from the checker's 10x saturation cap).
CASES = [
    (16, 120, 160, 20000, 1, False, 1.4, 110.0, 20),
    (18, 180, 230, 25000, 1, False, 1.4, 105.0, 20),
    (20, 240, 300, 30000, 1, False, 1.4, 100.0, 20),
    (20, 300, 370, 35000, 1, False, 1.4, 95.0, 20),
    (22, 260, 340, 40000, 800, True, 6, 20.0, 15),
    (24, 300, 400, 50000, 1000, True, 6, 19.0, 15),
    (26, 340, 460, 60000, 1300, True, 6, 18.0, 14),
    (28, 380, 520, 70000, 1500, True, 7, 18.0, 14),
    (30, 420, 580, 80000, 1800, True, 7, 17.0, 13),
    (32, 460, 640, 90000, 2000, True, 8, 17.0, 13),
]


def gen_case(test_id: int):
    nbins, flo, fhi, pfs, nfloor, wideband, fmax_mult, budget_mult, wfilt_div = \
        CASES[test_id - 1]
    rnd = random.Random(3000 + test_id * 6151)

    bins = []
    freqs = set()

    def add(f, p):
        while f in freqs:
            f += 1
        freqs.add(f)
        bins.append((f, p))

    nin = max(2, nbins // 3)
    for _ in range(nin):
        f = rnd.randint(flo, fhi)
        p = rnd.randint(pfs // 20, pfs // 6)
        add(f, p)

    nout = nbins - nin
    if wideband:
        fmax = int(fhi * fmax_mult)
        for _ in range(nout):
            f = rnd.randint(fhi + 1, fmax)
            p = rnd.randint(pfs // 20, pfs // 6)
            add(f, p)
    else:
        # warm-up cases: interferers strictly BELOW the band only, so even a
        # forced-low sample rate (as long as it still clears FHI) never lets
        # them alias in regardless of filter order -- these cases are meant
        # to be gentle on the naive "maximize resolution" recipe.
        fmin = max(1, flo // 3)
        fmax = max(fmin + 1, flo - 1)
        for _ in range(nout):
            f = rnd.randint(fmin, fmax)
            p = rnd.randint(pfs // 40, pfs // 10)
            add(f, p)

    budget = int(round(fhi * budget_mult))
    wfilt = max(1, budget // wfilt_div)
    return nbins, budget, wfilt, pfs, nfloor, flo, fhi, bins


def main():
    test_id = int(sys.argv[1])
    nbins, budget, wfilt, pfs, nfloor, flo, fhi, bins = gen_case(test_id)
    lines = [f"{nbins} {budget} {wfilt} {pfs} {nfloor} {flo} {fhi}"]
    for f, p in bins:
        lines.append(f"{f} {p}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

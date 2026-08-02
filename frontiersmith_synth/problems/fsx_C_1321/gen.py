#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE catalyst-site-placement instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small -> large.

Family: catalyst-site-place. Mechanisms composed into the objective:
  - site-site-interaction: active sites within r_screen cells of each other
    crowd each other's usable capacity down multiplicatively.
  - diffusion-limited-supply: reactant only reaches a site by explicit
    finite-difference diffusion from two fixed-concentration boundary
    reservoirs -- a site can only convert what has already diffused in.
  - poisoning-accumulation: each site's capacity decays with its own
    cumulative turnover (poison never clears).

Trap (tests 4..10, "diffusion-limited", 7 of 10): D is drawn small, so the
diffusion length is short relative to L. A placement that maximizes NOMINAL
site count by packing everything into a dense cluster near a boundary (or
splitting the budget between both boundaries -- the naive "put sites where
supply is closest" heuristic) starves the interior of the cluster of fresh
reactant and pays the full crowding penalty on every close pair, while a
placement that SPACES its sites out (closer to the diffusion length) lets
each site tap a largely unshared patch of the strip. Tests 1..3 ("fast
diffusion" warm-ups) use large D, where the boundary-anchored cluster keeps
up fairly well (diffusion refills faster than any single placement can
starve it) -- so the gap between the naive heuristic and spacing is small
there and large everywhere else (>=3 of 10, in fact 7 of 10, required by the
brief).
"""
import sys
import random

SIZES = {1: 20, 2: 22, 3: 24, 4: 26, 5: 28, 6: 30, 7: 34, 8: 38, 9: 42, 10: 46}
FAST_IDS = {1, 2, 3}


def build(tid):
    rng = random.Random(90000 + 131 * tid)
    L = SIZES[tid]

    if tid in FAST_IDS:
        D = rng.uniform(2.0, 3.2)
    else:
        # diffusion length shrinks further relative to L as the ladder climbs
        D = rng.uniform(0.05, 0.30) * (26.0 / L)

    v_max = rng.uniform(0.35, 0.65)
    gamma = rng.uniform(0.15, 0.35)
    r_screen = rng.randint(2, 3)
    poison_rate = rng.uniform(0.03, 0.08)
    C0 = rng.uniform(0.85, 1.15)
    dens = rng.uniform(0.28, 0.36)
    B = max(4, round(L * dens))
    B = min(B, int(0.4 * L))
    B = max(4, B)

    lines = [f"{L} {B}",
             f"{D:.6f} {v_max:.6f} {gamma:.6f} {r_screen} {poison_rate:.6f} {C0:.6f}"]
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()

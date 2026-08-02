#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE 'pollution-attribution-wind' instance to stdout.

Deterministic: every draw is seeded ONLY from testId (via common.build_world /
common.SECRET_BASE and the local TRAIN_BASE stream below).

WORLD (never printed): K candidate factory coordinates, a SECRET true emission
rate per factory, and a per-instance background-trend law. Some factories sit in
tight BEARING CLUSTERS as seen from the receptor (2-3 factories whose direction
from the receptor differs by only a few degrees) -- on any SINGLE day their
plume contributions are nearly indistinguishable; only days whose wind direction
differs enough across the log can ever separate them.

TRAIN LOG (printed): a mix of
  - "prevailing" days: low wind speed (broad, near-isotropic plume -> everything
    blends together) blowing from the direction of this instance's LARGEST
    bearing cluster. Low speed means poor dispersal, so these are the HIGHEST
    measured concentrations -- and simultaneously the LEAST informative days for
    telling that cluster's members apart.
  - "informative" days: higher wind speed (sharp, narrow plume) with directions
    that vary across the log, including some aimed squarely at individual
    cluster members. These carry the separating signal but are diluted (lower
    magnitude) by the same wind that sharpens them.
Row PRINT ORDER is shuffled relative to day_id (temporal alignment: a solver
must key off the explicit day_id field, not row position, when applying the
background trend).
"""
import sys, random
import common


def build_train(test_id):
    world = common.build_world(test_id)
    K = world["K"]
    trap = world["trap"]
    rng = random.Random(9_133_000 + 131 * test_id)
    noise_rng = random.Random(9_133_000 + 727 * test_id)

    # clean, sharp "anchor" days per source -- guarantees every source (cluster
    # member or standalone) is the dominant contributor on several days, so no
    # source is ever structurally invisible OR left to a single noisy sighting.
    base_revisits = 2
    plan = []
    for i in range(K):
        for _ in range(base_revisits):
            plan.append(("anchor", world["all_bearings"][i]))
    # cluster members get EXTRA anchor revisits on top of the base (redundancy
    # against noise) -- more on trap cases, since averaging several noisy-but-
    # clean sightings of a tight cluster member is exactly what the "trust
    # informative days" insight can exploit and a magnitude-weighted fit
    # (drowned out by the prevail flood) cannot.
    extra_revisits = 3 if trap else 1
    for member_idx in world["cluster_member_idx"]:
        for i in member_idx:
            for _ in range(extra_revisits):
                plan.append(("anchor", world["all_bearings"][i]))
    # a handful of broad, moderate-speed days for general robustness
    n_broad = 1 if trap else 3
    plan += [("info_broad", None)] * n_broad
    # the "loud but uninformative" stagnant days -- outnumber the anchors when this
    # is a trap case (heavily, by RAW COUNT), so a magnitude-weighted heuristic's
    # normal equations are dominated by the near-collinear cluster-confounding rows.
    prevail_mult = 10.0 if trap else 3.0
    n_prevail = max(3, round(prevail_mult * K))
    plan += [("prevail", None)] * n_prevail

    D = len(plan)
    day_ids = list(range(1, D + 1))
    rows = []
    for day_id, (kind, target) in zip(day_ids, plan):
        wd, ws = common.sample_day(rng, kind, world, target_bearing=target)
        y = common.concentration(world, day_id, wd, ws, noise_rng)
        rows.append((day_id, wd, ws, y))
    rng.shuffle(rows)  # print order != day_id order
    world["D_train"] = D
    return world, rows


def emit(test_id):
    world, rows = build_train(test_id)
    K, D = world["K"], world["D_train"]
    out = [f"{test_id} {K} {D}"]
    out.append("%.6f %.6f %d %.6f %.6f %.6f %.6f" % (
        world["A0"], world["A1"], world["P"],
        common.SIGMA_MAX_DEG, common.ALPHA, common.L0, common.BETA))
    for (sx, sy) in world["sources"]:
        out.append("%.3f %.3f" % (sx, sy))
    for (day_id, wd, ws, y) in rows:
        out.append("%d %.4f %.4f %.6f" % (day_id, wd, ws, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    emit(tid)

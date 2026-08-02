# TIER: strong
"""The insight: reformulate item selection as maximizing information MINUS a
penalty on projected exposure, instead of maximizing information alone. This
is not "greedy plus a knob" picked by trial and error -- the penalty WEIGHT is
DERIVED from the instance's own structure, and the forecast is deliberately
under-trusted:

1. Information and exposure live on different scales, so the penalty weight
   must track the item bank's own information scale, not a fixed magic
   number. The max Fisher information any 2PL item can contribute (at
   P=0.5) is a^2/4, so we scale the exposure penalty by the BANK-AVERAGE of
   that quantity -- a bank of weaker items gets a gentler penalty than a bank
   of highly discriminating ones, keeping the info-vs-exposure trade-off in
   the same units on every instance.

2. That scale is then divided by how tight the cap actually is (mean
   cap_frac across items): a loose cap barely needs throttling, a tight one
   needs a much stronger penalty to avoid ever reaching it. This is the
   exchange argument the naive recipe never makes: sacrifice a little
   per-decision information now in exchange for the item still being usable
   (and un-leaked) for the rest of the season.

3. The penalty is CONVEX in projected exposure (exponent 2): it barely bites
   until an item is meaningfully in demand, then ramps up sharply as it
   approaches its cap -- exactly the shape needed to leave calm, low-demand
   items untouched while decisively throttling hot ones.

4. Projected exposure blends the supplied population-agnostic forecast with
   this season's OWN live exposure counts, weighted only 15% toward the
   forecast. The forecast is useful only as a cold-start prior (before any
   live signal exists) -- it assumes a generic, ability-spread-out reference
   population, so trusting it heavily is itself a trap: whenever the real
   season's true ability concentrates somewhere the forecast under-predicted,
   over-relying on it fails to throttle the items that actually turn out to
   be hot. Once live counts accumulate they are ground truth for THIS season
   and should dominate."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    items = inst["items"]
    avg_info_scale = sum((float(it["a"]) ** 2) / 4.0 for it in items) / len(items)
    cap_frac_mean = sum(float(it["cap_frac"]) for it in items) / len(items)

    K = 0.5  # trade-off constant: how many "information units" one unit of
             # projected exposure is worth once scaled into the bank's own units
    exposure_weight = K * avg_info_scale / max(cap_frac_mean, 0.05)

    ans = {
        "info_weight": 1.0,
        "exposure_weight": exposure_weight,
        "exposure_shape": 2.0,
        "hint_trust": 0.15,
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()

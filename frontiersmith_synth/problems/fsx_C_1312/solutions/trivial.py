# TIER: trivial
"""Non-adaptive baseline: ignore both information and exposure entirely
(info_weight=0, exposure_weight=0). With every score tied at 0, the evaluator's
deterministic tie-break (smallest item index first) turns this into a fixed
round-robin item order -- exposure ends up almost perfectly even across the
bank (nobody targets the same handful of items), but items are never matched
to the examinee's ability, so estimates are mediocre everywhere. This
reproduces the evaluator's own internal weak baseline (obj_base), so it
anchors to r ~ 0.1 on every instance."""
import sys, json


def main():
    json.load(sys.stdin)  # public instance unused by this policy
    ans = {"info_weight": 0.0, "exposure_weight": 0.0,
           "exposure_shape": 1.0, "hint_trust": 0.0}
    print(json.dumps(ans))


if __name__ == "__main__":
    main()

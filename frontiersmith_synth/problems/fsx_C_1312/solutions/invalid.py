# TIER: invalid
"""Broken policy: 'exposure_weight' is a non-finite value (Infinity), which
JSON can print (Python's json.dumps allows it by default) but is illegal for
the evaluator's strict answer validation (must be finite) -- every instance is
rejected -> scores 0."""
import sys, json


def main():
    json.load(sys.stdin)
    ans = {"info_weight": 1.0, "exposure_weight": float("inf"),
           "exposure_shape": 2.0, "hint_trust": 0.1}
    print(json.dumps(ans))


if __name__ == "__main__":
    main()

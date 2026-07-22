# TIER: trivial
# Do nothing: release 0 every day. This is exactly the evaluator's own weak-baseline
# construction, so it scores ~0.1. The level only ever rises (inflow is never negative)
# until it hits capacity, after which every further drop of rain spills in full at the
# flood penalty -- shortage never triggers here since the level never falls.
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    print(json.dumps({"releases": [0.0] * T}))


if __name__ == "__main__":
    main()

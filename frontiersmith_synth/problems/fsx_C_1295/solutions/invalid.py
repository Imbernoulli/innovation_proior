# TIER: invalid
"""Malformed submission: priority is not a permutation (robot 0 repeated,
robot R-1 missing) -- rejected on every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    R = inst["R"]
    priority = [0] * R  # not a permutation -> invalid
    routes = [[] for _ in range(R)]
    print(json.dumps({"priority": priority, "routes": routes}))


if __name__ == "__main__":
    main()

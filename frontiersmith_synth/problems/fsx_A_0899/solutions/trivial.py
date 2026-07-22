# TIER: trivial
"""Reject every truck: keep only the free backbone."""
import sys, json


def main():
    json.load(sys.stdin)
    print(json.dumps({"accept": False, "state": None}))


if __name__ == "__main__":
    main()

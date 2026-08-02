# TIER: trivial
"""Naive reference: release exactly today's demand every day, clipped to
nothing else. No storage management, no forecast use, no flood awareness
at all -- water simply accumulates until it eventually spills."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    demand = inst["demand"]
    release = [float(d) for d in demand]
    print(json.dumps({"release": release}))


if __name__ == "__main__":
    main()

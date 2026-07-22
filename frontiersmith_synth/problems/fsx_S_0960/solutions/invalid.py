# TIER: invalid
"""Always claims the origin cell for every order, never checking for overlap.
The first order is (usually) legal alone; the second collides with the first
order's own kerf-locked footprint, making the whole cutting plan illegal."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]
    out = [{"action": "place", "x": 0, "y": 0, "rot": 0} for _ in range(n)]
    print(json.dumps({"decisions": out}))


if __name__ == "__main__":
    main()

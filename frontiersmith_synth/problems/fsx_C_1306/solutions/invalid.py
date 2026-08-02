# TIER: invalid
"""Malformed candidate: wrong-length, partially non-numeric release list."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    release = [1.0] * (T - 2)      # wrong length
    release.append("lots")          # also not a number
    print(json.dumps({"release": release}))


if __name__ == "__main__":
    main()

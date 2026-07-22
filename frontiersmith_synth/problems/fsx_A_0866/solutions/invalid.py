# TIER: invalid
# Ignores the u_max box constraint entirely -- every instance must score 0.
import sys
import json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    u_max = inst["u_max"]
    print(json.dumps({"u": [u_max * 1000.0] * T}))


if __name__ == "__main__":
    main()

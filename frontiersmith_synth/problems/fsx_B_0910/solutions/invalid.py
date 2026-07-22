# TIER: invalid
# Always emits an out-of-range symbol -> every instance is rejected as infeasible.
import sys, json


def main():
    inst = json.load(sys.stdin)
    L, A = inst["N"], inst["A"]
    print(json.dumps({"guess": [A] * L}))


main()

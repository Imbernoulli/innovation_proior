# TIER: invalid
# Deliberately malformed: wrong grid shape (T+1 columns) and an out-of-range
# exit index -- must be rejected and scored 0.0 on every instance.
import sys, json


def main():
    inst = json.load(sys.stdin)
    Z, T, E = inst["n_zones"], inst["T"], inst["n_exits"]
    guidance = [[E] * (T + 1) for _ in range(Z)]
    print(json.dumps({"guidance": guidance}))


main()

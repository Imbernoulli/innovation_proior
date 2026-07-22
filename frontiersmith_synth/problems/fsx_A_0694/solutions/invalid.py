# TIER: invalid
# Emits a sequence one entry too short (T-1 entries instead of T). Fails the
# evaluator's length check -> the instance scores 0.0.
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    seq = [0] * max(0, T - 1)
    print(json.dumps({"sequence": seq}))


if __name__ == "__main__":
    main()

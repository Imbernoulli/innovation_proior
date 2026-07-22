# TIER: invalid
# Emits a releases array one entry too short (T-1 instead of T). Fails the evaluator's
# length check -> every instance scores 0.0.
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    print(json.dumps({"releases": [0.0] * max(0, T - 1)}))


if __name__ == "__main__":
    main()

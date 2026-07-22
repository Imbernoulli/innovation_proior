# TIER: greedy
import sys, json


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]
    K = inst["K"]
    # Textbook cyclic coordinate descent: sweep every gauge in index order,
    # wrap around, keep going until the budget is spent. No use of the
    # coupling graph at all -- every gauge is treated identically.
    order = [i % n for i in range(K)]
    print(json.dumps({"order": order}))


main()

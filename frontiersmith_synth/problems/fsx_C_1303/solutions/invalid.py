# TIER: invalid
# Malformed answer: allocation rows are the wrong length and the boost pool is
# over-spent, and the buffer list is short -- must be rejected (score 0) by
# the evaluator's strict validation.
import sys, json


def main():
    inst = json.load(sys.stdin)
    K, T, P = inst["K"], inst["T"], inst["P"]
    # wrong row width (K+1 instead of K) and blows the boost budget
    alloc = [[P] * (K + 1) for _ in range(T)]
    # too few buffers
    buffers = [1]
    print(json.dumps({"alloc": alloc, "buffers": buffers}))


if __name__ == "__main__":
    main()

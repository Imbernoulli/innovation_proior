# TIER: invalid
# Plausible-looking but infeasible: emits the wrong number of cuts (three
# extra), which violates the fixed budget B and must score 0.
import sys, json


def main():
    inst = json.load(sys.stdin)
    N = inst["N"]
    B = inst["B"]
    step = N // (B + 3)
    cuts = [step * k for k in range(1, B + 3)]  # len == B+2, should be B-1
    print(json.dumps({"cuts": cuts}))


main()

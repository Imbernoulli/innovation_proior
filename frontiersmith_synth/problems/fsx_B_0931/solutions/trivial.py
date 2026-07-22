# TIER: trivial
# Uniform grid: ignore the probe data entirely, cut the domain into B
# equal-width cells. This is the reference construction the grader itself
# normalizes against.
import sys, json


def main():
    inst = json.load(sys.stdin)
    N = inst["N"]
    B = inst["B"]
    step = N // B
    cuts = [step * k for k in range(1, B)]
    print(json.dumps({"cuts": cuts}))


main()

# TIER: invalid
import sys, json


def main():
    inst = json.load(sys.stdin)
    N = inst["n"]
    # deliberately out-of-range palette index -> evaluator must reject with score 0
    print(json.dumps({"grid": [[999 for _ in range(N)] for _ in range(N)]}))


main()

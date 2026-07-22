# TIER: invalid
import sys, json

def main():
    inst = json.load(sys.stdin)
    R, C, K = inst["R"], inst["C"], inst["K"]
    # out-of-range tile indices everywhere -> must be rejected, score 0
    types = [[K + 5] * C for _ in range(R)]
    print(json.dumps({"types": types}))

main()

# TIER: trivial
import sys, json

def main():
    inst = json.load(sys.stdin)
    R, C, K = inst["R"], inst["C"], inst["K"]
    t = K // 2
    types = [[t] * C for _ in range(R)]
    print(json.dumps({"types": types}))

main()

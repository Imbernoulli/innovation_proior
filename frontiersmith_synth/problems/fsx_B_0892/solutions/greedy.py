# TIER: greedy
import sys, json

def main():
    inst = json.load(sys.stdin)
    R, C = inst["R"], inst["C"]
    eff_table = inst["eff_table"]
    target = inst["target"]
    types = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            tv = target[r][c]
            best_t = min(range(len(eff_table)), key=lambda t: abs(eff_table[t] - tv))
            types[r][c] = best_t
    print(json.dumps({"types": types}))

main()

# TIER: invalid
# Emits a netlist with an empty bottom resistance on every tap (zero counts) ->
# infeasible divider (no bottom leg). Checker must score 0.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    K = int(next(it)); M = int(next(it))
    lines = []
    for _ in range(K):
        top = [0] * M; bot = [0] * M
        top[0] = 1   # top present, bottom all zero -> infeasible
        lines.append(" ".join(map(str, top + bot)))
    sys.stdout.write("\n".join(lines) + "\n")

main()

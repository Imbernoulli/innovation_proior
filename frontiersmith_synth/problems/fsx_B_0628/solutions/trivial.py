# TIER: trivial
# Coast the whole race: minimum intensity every lap, never pit.
# This reproduces the checker's internal baseline -> Ratio ~= 0.1.
import sys

def main():
    d = sys.stdin.read().split()
    L = int(d[0])  # k not needed
    out = []
    for _ in range(L):
        out.append("0 0")
    sys.stdout.write("\n".join(out) + "\n")

main()

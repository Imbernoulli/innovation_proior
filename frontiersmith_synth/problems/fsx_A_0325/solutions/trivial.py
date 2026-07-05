# TIER: trivial
# Reproduces the checker's baseline: pour uniform intensity into the front
# half of the arena, leave the rest dark.  c1 ~= 4  ->  Ratio ~= 0.1.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])
    L = (n + 1) // 2
    f = [1] * L + [0] * (n - L)
    sys.stdout.write(" ".join(map(str, f)) + "\n")

main()

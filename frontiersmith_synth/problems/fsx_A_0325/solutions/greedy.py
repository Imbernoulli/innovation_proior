# TIER: greedy
# Spread the mass as evenly as possible: uniform intensity across all n slots.
# The self-echo becomes a single centered triangle, giving c1 = 2 exactly --
# the strong local optimum that every smooth/analytic reshaping fails to beat.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])
    f = [1] * n
    sys.stdout.write(" ".join(map(str, f)) + "\n")

main()

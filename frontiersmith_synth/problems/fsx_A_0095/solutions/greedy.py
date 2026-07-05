# TIER: greedy
# Uniform (flat) production: bake V loaves every shift. A smarter-than-tent
# heuristic (a constant sequence gives c = 2 exactly, beating the triangular
# baseline) but far from optimal.
import sys

def main():
    n, V = map(int, sys.stdin.read().split())
    a = [V] * n
    sys.stdout.write(" ".join(map(str, a)) + "\n")

if __name__ == "__main__":
    main()

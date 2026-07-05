# TIER: invalid
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    # deliberately infeasible: emit the token '2' (not a valid 0/1 depth) -> must score 0
    sys.stdout.write(" ".join("2" for _ in range(n)) + "\n")

main()

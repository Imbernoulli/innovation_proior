# TIER: invalid
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0]); D = int(d[2])
    # deliberately infeasible: emit a config equal to D (out of range) -> checker must score 0
    vals = ["0"] * n
    if n > 0:
        vals[0] = str(D)
    sys.stdout.write(" ".join(vals) + "\n")

main()

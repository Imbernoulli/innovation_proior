# TIER: invalid
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    D = int(d[2])
    # deliberately infeasible: emit a channel above D -> checker must score 0
    vals = ["1"] * n
    if n > 0:
        vals[0] = str(D + 1)
    sys.stdout.write(" ".join(vals) + "\n")

main()

# TIER: invalid
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    # deliberately infeasible: emit a value outside {0,1} -> checker must score 0
    vals = ["0"] * n
    if n > 0:
        vals[0] = "2"
    sys.stdout.write(" ".join(vals) + "\n")

main()

# TIER: invalid
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    # deliberately infeasible: claim to install 1 pump but give an out-of-range index
    # -> checker rejects (index out of range) -> Ratio 0.0
    sys.stdout.write("1\n")
    sys.stdout.write("%d\n" % (n + 1))

main()

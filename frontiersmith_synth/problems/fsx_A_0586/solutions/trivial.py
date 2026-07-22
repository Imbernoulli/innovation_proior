# TIER: trivial
# Identity matching (drone i -> target point i as listed), everyone in wave 0.
# This reproduces the checker's internal baseline B  ->  Ratio ~= 0.1.
import sys

def main():
    d = sys.stdin.read().split()
    N = int(d[0])
    out = ["%d 0" % i for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")

main()

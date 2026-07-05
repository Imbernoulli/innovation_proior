# TIER: trivial
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    # install a pump in every tank -> reproduces the checker baseline B -> Ratio = 0.1
    sys.stdout.write("%d\n" % n)
    sys.stdout.write(" ".join(str(v) for v in range(1, n + 1)) + "\n")

main()

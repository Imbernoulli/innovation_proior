# TIER: trivial
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    # all wells in PRODUCTION mode (0) -> reproduces the checker's baseline B -> Ratio = 0.1
    sys.stdout.write(" ".join("0" for _ in range(n)) + "\n")

main()

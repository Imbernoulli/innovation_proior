# TIER: trivial
import sys

def main():
    d = sys.stdin.buffer.read().split()
    n = int(d[0])
    # every station on channel 1 -> reproduces the checker's baseline B -> Ratio = 0.1
    sys.stdout.write(" ".join("1" for _ in range(n)) + "\n")

main()

# TIER: invalid
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0])
    # emit an explicit raiding line: 0^n, 1^n, 2^n  (each coord sums to 0+1+2=3==0 mod 3)
    line = ["0" * n, "1" * n, "2" * n]
    print(len(line))
    print("\n".join(line))

main()

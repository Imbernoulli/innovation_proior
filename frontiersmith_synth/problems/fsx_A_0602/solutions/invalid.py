# TIER: invalid
# Emits infeasible output (negative prices) -> checker must score 0.
import sys

def main():
    data = sys.stdin.buffer.read().split()
    M = int(data[0])
    sys.stdout.write("\n".join("-5" for _ in range(M)) + "\n")

main()

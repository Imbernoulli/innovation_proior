# TIER: invalid
# Emits an infeasible artifact: a move index far outside the legal range
# [2, N-1]. The checker must reject it and score 0.
import sys

def main():
    sys.stdin.read()
    print(1)
    print("s 999999")

if __name__ == "__main__":
    main()

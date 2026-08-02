# TIER: trivial
"""Keep every reaction. Always feasible (zero deviation) -- the checker's own baseline."""
import sys


def main():
    tokens = sys.stdin.read().split()
    ip = 0
    n = int(tokens[ip]); ip += 1
    m = int(tokens[ip]); ip += 1
    # target, P, T_horizon, N_steps, epsilon -- unused by this tier
    print(m)
    print(" ".join(str(i) for i in range(m)))


if __name__ == "__main__":
    main()

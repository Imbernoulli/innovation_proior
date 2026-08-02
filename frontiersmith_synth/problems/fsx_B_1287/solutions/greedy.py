# TIER: greedy
"""The obvious textbook approach: rebalance to exact delta-neutrality at every
single timestep. Locally always "correct", but pays full transaction cost on
every step -- including tiny noise wiggles and post-jump whipsaws that partially
revert on their own the very next step."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    idx += N + 1                                  # skip S
    D = [float(toks[idx + i]) for i in range(N + 1)]; idx += N + 1
    # (skip G and cost line -- greedy ignores gamma and cost entirely)
    print(" ".join("%.10g" % D[t] for t in range(1, N + 1)))


if __name__ == "__main__":
    main()

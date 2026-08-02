# TIER: greedy
"""The industry-standard textbook approach: a LINEAR AGE-BASED GLIDEPATH.
w depends only on how far through the horizon we are -- start risky,
de-risk in a straight line to a safe end weight -- and completely ignores
the funded ratio / which bucket the fund is actually in. This is exactly
the "obvious first pass" the problem is designed to make into a trap: on
calm paths it looks fine, but it neither locks in gains after an early
rally pushes the fund into overfunded territory, nor keeps taking risk
when the fund is still underfunded with time left to recover."""
import sys

W_START = 0.85
W_END = 0.20


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0])
    lines = []
    for t in range(1, T + 1):
        if T <= 1:
            w = 0.5
        else:
            w = W_START - (W_START - W_END) * (t - 1) / (T - 1)
        w = min(1.0, max(0.0, w))
        lines.append(" ".join(f"{w:.6f}" for _ in range(5)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

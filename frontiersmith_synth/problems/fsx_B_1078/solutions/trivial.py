# TIER: trivial
"""Reproduces the checker's own baseline: the top min(k,N) individually most
profitable tasks, each scheduled alone on its own production line. Ignores all
precedence structure entirely (singleton lines are always feasible)."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    N = nxt(); M = nxt(); k = nxt(); H = nxt()
    profit = [0] * (N + 1)
    for i in range(1, N + 1):
        profit[i] = nxt()
    # skip edges, unused
    p += 2 * M

    order = sorted(range(1, N + 1), key=lambda t: (-profit[t], t))
    chosen = order[:min(k, N)]

    out = [str(len(chosen))]
    for t in chosen:
        out.append(f"1 {t}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

# TIER: greedy
"""The obvious first instinct: mass = damping, so pile as much mass as the cap
allows directly onto the cells you want quiet, in the order given, then spread
any leftover budget evenly over the rest. Never looks at the spectrum at all,
so it has no idea that this can flip which mode ends up ranked k."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    k = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1
    budget = int(toks[p]); p += 1
    t = int(toks[p]); p += 1
    targets = [int(toks[p + i]) for i in range(t)]
    p += t

    Nc = N * N
    loads = [0] * Nc
    remaining = budget

    for c in targets:
        if remaining <= 0:
            break
        add = min(cap, remaining)
        loads[c] = add
        remaining -= add

    if remaining > 0:
        others = [i for i in range(Nc) if i not in targets]
        if others:
            base = remaining // len(others)
            rem = remaining - base * len(others)
            for j, i in enumerate(others):
                add = min(cap - loads[i], base + (1 if j < rem else 0))
                add = max(0, add)
                loads[i] += add
                remaining -= add

    print(" ".join(map(str, loads)))


if __name__ == "__main__":
    main()

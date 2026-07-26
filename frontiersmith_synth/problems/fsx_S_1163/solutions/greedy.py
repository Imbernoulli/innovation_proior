# TIER: greedy
"""The 'obvious' quality-greedy recipe: sort ALL bookable pointings by their raw
quality weight (descending), then take the highest-weight pointings, skipping any
whose night is already used, until the B-night budget is spent. This ignores which
narrative pairs each pointing actually splits -- it is a plausible first instinct
given a field literally called 'weight', and it is the trap: low-weight pointings
that are the ONLY way to split a hard family's confusable cluster get starved out
by abundant higher-weight pointings on other nights."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)

    def nxt():
        return next(it)

    N = int(nxt()); D = int(nxt()); K = int(nxt()); B = int(nxt()); T = int(nxt())
    slots = []  # (sid, night, weight)
    for _ in range(T):
        sid = int(nxt()); night = int(nxt()); w = int(nxt()); ns = int(nxt())
        for _ in range(ns):
            nxt()
        slots.append((sid, night, w))
    # families irrelevant to this recipe.

    slots.sort(key=lambda t: (-t[2], t[0]))  # weight desc, slot id asc (determinism)
    used_nights = set()
    chosen = []
    for sid, night, w in slots:
        if len(chosen) >= B:
            break
        if night in used_nights:
            continue
        used_nights.add(night)
        chosen.append(sid)

    print(len(chosen))
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()

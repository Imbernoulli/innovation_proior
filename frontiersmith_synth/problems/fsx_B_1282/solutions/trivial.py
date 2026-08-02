# TIER: trivial
"""Reproduces the checker's own naive baseline: rank participant-windows by raw
total CANCEL COUNT (ignores side, ignores timing) and flag the top-K within the
alert budget. This is the textbook "cancel rate is suspicious" instinct with no
insight at all -- it is exactly the trap the family's innovation_hook warns
about (a market maker cancels just as much, just on both sides)."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    test_id = int(data[idx].strip()); idx += 1
    N, W, K = map(int, data[idx].split()); idx += 1
    E = int(data[idx].strip()); idx += 1

    cancel_count = {}
    for i in range(E):
        toks = data[idx + i].split()
        w, pid, t, side, action, size = toks
        w = int(w); pid = int(pid); action = action
        if action == "C":
            key = (w, pid)
            cancel_count[key] = cancel_count.get(key, 0) + 1
    idx += E

    all_pw = [(w, pid) for w in range(W) for pid in range(N)]
    all_pw.sort(key=lambda pw: (-cancel_count.get(pw, 0), pw[0], pw[1]))
    flagged = all_pw[:K]

    out = [str(len(flagged))]
    for (w, pid) in flagged:
        out.append(f"{w} {pid}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

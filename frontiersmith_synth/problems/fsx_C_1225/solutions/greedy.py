# TIER: greedy
"""The obvious textbook move: give every distinct key its own exact token
bucket, sized by how much traffic it sent, and treat everyone "fairly" --
no attention to the memory budget, no attention to which keys are abusive.
This is optimal on small warm-up traces and blows the memory budget (or, when
it happens to fit, admits abusive heavy hitters just as generously as good
ones) whenever the key population is large."""
import sys

RATE_MAX = 5000


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); M = int(next(it))
    next(it); next(it); next(it)  # A, B, P (hash constants) -- unused, we go per-key
    R = int(next(it))

    count = {}
    for _ in range(R):
        t = int(next(it)); key = int(next(it)); lab = int(next(it))
        c = count.get(key)
        if c is None:
            count[key] = [0, 0]
            c = count[key]
        c[0] += 1  # total volume
        c[1] += t  # (unused, kept simple)

    keys = sorted(count.keys())
    H = len(keys)
    lines = [f"{H} 1"]
    for k in keys:
        vol = count[k][0]
        cap = min(RATE_MAX, vol)
        rate = max(1, min(RATE_MAX, vol // T + 1))
        lines.append(f"{k} {cap} {rate}")
    lines.append("8 2")  # fallback group template (essentially unused: H covers everyone)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

# TIER: trivial
# The checker's own baseline: one full-window (s=0, d=T, p=Pmax) firing per
# DISTINCT nearest burner needed to cover the targets. No timing/power tuning,
# no consolidation beyond the trivial nearest-burner dedup. Reproduces the
# checker's internal baseline B exactly -> Ratio ~= 0.1.
import sys


def nearest_burner_idx(pos, burners):
    best_i, best_d = 0, None
    for i, bp in enumerate(burners):
        dd = abs(pos - bp)
        if best_d is None or dd < best_d or (dd == best_d and i < best_i):
            best_d = dd
            best_i = i
    return best_i


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it)); H = int(next(it))
    Pmax = int(next(it)); F0 = int(next(it))
    numB = int(next(it)); numT = int(next(it))
    burners = [int(next(it)) for _ in range(numB)]
    targets = [int(next(it)) for _ in range(numT)]

    used = sorted({nearest_burner_idx(x, burners) for x in targets})
    firings = [(bi, 0, T, Pmax) for bi in used]

    out = [str(len(firings))]
    for (bi, s, d, p) in firings:
        out.append(f"{bi} {s} {d} {p}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

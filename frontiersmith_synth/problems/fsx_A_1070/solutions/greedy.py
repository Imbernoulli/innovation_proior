# TIER: greedy
# The "obvious" per-zone reflex: for EACH target independently, assign its
# nearest burner and fire immediately (start=0), using the shortest duration a
# naive, distance-blind, no-diffusion-loss estimate says should be enough:
#   d = ceil(H / Pmax)
# This ignores that heat needs travel TIME to cross distance to the target and
# that a firing which stops early lets heat leak away (redistribute sideways)
# before the exact-time check at T. It also never deduplicates burners shared
# by several targets, paying redundant setup fuel.
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

    d_naive = min(T, -(-H // Pmax))  # ceil(H/Pmax), never exceeding the deadline
    firings = []
    for x in targets:
        bi = nearest_burner_idx(x, burners)
        firings.append((bi, 0, d_naive, Pmax))

    out = [str(len(firings))]
    for (bi, s, d, p) in firings:
        out.append(f"{bi} {s} {d} {p}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

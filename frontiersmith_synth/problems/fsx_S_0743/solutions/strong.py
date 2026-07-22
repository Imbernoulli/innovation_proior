# TIER: strong
# Insight: two given anchors that lie in the same orbit under the 8 symmetries
# will BOTH get inked as soon as either one is drawn and symmetrized -- so the
# fundamental domain only needs ONE seed cell per underlying orbit, not one per
# given anchor. Dedup the anchor list by orbit membership (an anchor is
# redundant if some already-kept representative maps onto it under one of the
# 8 symmetries) before building the macro. Same macro+8-call skeleton as
# greedy, but the macro body only pays for the true generating set -- exactly
# the "choose WHAT to draw" compression the checker's B (orbit closure size)
# rewards.
import sys


def apply_t(t, x, y, N):
    if t == 0: return (x, y)
    if t == 1: return (N - 1 - y, x)
    if t == 2: return (N - 1 - x, N - 1 - y)
    if t == 3: return (y, N - 1 - x)
    if t == 4: return (N - 1 - x, y)
    if t == 5: return (x, N - 1 - y)
    if t == 6: return (y, x)
    if t == 7: return (N - 1 - y, N - 1 - x)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); A = int(next(it))
    anchors = [(int(next(it)), int(next(it))) for _ in range(A)]

    kept = []
    for a in anchors:
        redundant = False
        for (rx, ry) in kept:
            for t in range(8):
                if apply_t(t, rx, ry, N) == a:
                    redundant = True
                    break
            if redundant:
                break
        if not redundant:
            kept.append(a)

    out = ["DEF 0"]
    for (x, y) in kept:
        out.append("DOT %d %d" % (x, y))
    out.append("END")
    for g in range(8):
        out.append("CALL 0 %d" % g)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

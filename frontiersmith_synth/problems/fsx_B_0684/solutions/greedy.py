# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); C = int(next(it)); R = int(next(it))
    target = [int(next(it)) for _ in range(n)]
    tools = []
    for _ in range(m):
        mx = int(next(it)); mn = int(next(it))
        tools.append((mx, mn))

    current = [R] * n
    # precedence-safe order: descending target, ties by ascending index -- the
    # "obvious" choice. Finishes ONE band completely before touching the next,
    # always taking the single deepest currently-legal pass.
    order = sorted(range(n), key=lambda i: (-target[i], i))

    passes = []
    last_tool = None
    for i in order:
        band = i + 1
        while current[i] > target[i]:
            headroom = current[i] - target[i]
            S = min(current[0:band])
            cap_chatter = (K * S) // band
            best_depth, best_tool = 0, None
            for tid, (mx, mn) in enumerate(tools, 1):
                cap = min(mx, headroom, cap_chatter)
                if cap < mn:
                    continue
                # naive "don't switch tools without a clear depth win" bias --
                # ties (or near-ties) favour whatever tool is already mounted.
                if cap > best_depth or (cap == best_depth and tid == last_tool):
                    best_depth, best_tool = cap, tid
            if best_tool is None:
                # should not happen (a minDepth==1 tool is always chatter-legal),
                # but guard defensively so the tape stays feasible.
                for tid, (mx, mn) in enumerate(tools, 1):
                    if mn == 1:
                        best_tool, best_depth = tid, 1
                        break
            passes.append((best_tool, band, best_depth))
            current[i] -= best_depth
            last_tool = best_tool

    out = [str(len(passes))]
    for (t, b, d) in passes:
        out.append(f"{t} {b} {d}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

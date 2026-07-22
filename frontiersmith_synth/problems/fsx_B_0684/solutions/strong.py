# TIER: strong
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
    passes = []
    last_tool = [None]

    finish_order = sorted(range(n), key=lambda i: (-target[i], i))

    def neighbours_ready(i):
        for j in (i - 1, i + 1):
            if 0 <= j < n and target[j] > target[i]:
                if current[j] != target[j]:
                    return False
        return True

    def best_move(i, tool_ids):
        band = i + 1
        headroom = current[i] - target[i]
        if headroom <= 0:
            return None
        S = min(current[0:band])
        cap_chatter = (K * S) // band
        best = None
        for tid in tool_ids:
            mx, mn = tools[tid - 1]
            cap = min(mx, headroom, cap_chatter)
            if cap < mn:
                continue
            depth = cap
            if depth == headroom and not neighbours_ready(i):
                depth -= 1
                if depth < mn:
                    continue
            if best is None or depth > best[0] or (depth == best[0] and tid == last_tool[0]):
                best = (depth, tid)
        return best

    # --- Phase A (rough, INTERLEAVED round-robin): every round, sweep bands in
    # descending axial distance so the far/deep bands get first claim on the
    # chatter budget while the rest of the rod is still close to full-thickness
    # -- but no band is ever fully depleted here unless finishing it right now
    # is already precedence-legal (only a genuinely BLOCKED final pass is
    # trimmed by 1 unit, so bands that don't actually need the reservation pay
    # no artificial tax). Repeated rounds let bands with light demand (shallow
    # target) finish quickly and unblock their deep neighbours mid-stream --
    # this is the interleaving that preserves stiffness exactly where deep cuts
    # are still pending, instead of exhausting one band before the next.
    rough_ids = [tid for tid, (mx, mn) in enumerate(tools, 1) if mn > 1]
    round_order = sorted(range(n), key=lambda i: -(i + 1))
    progress = True
    while progress:
        progress = False
        for i in round_order:
            mv = best_move(i, rough_ids)
            if mv is None:
                continue
            depth, tid = mv
            passes.append((tid, i + 1, depth))
            current[i] -= depth
            last_tool[0] = tid
            progress = True

    # --- Phase B (finish): whatever a band still needs after rough round-robin
    # is closed out in the precedence-safe global order, preferring the pure
    # finishing tool(s) so tool-changes stay batched.
    finish_ids = [tid for tid, (mx, mn) in enumerate(tools, 1) if mn == 1]
    if not finish_ids:
        finish_ids = list(range(1, m + 1))
    all_ids = list(range(1, m + 1))
    for i in finish_order:
        band = i + 1
        while current[i] > target[i]:
            mv = best_move(i, finish_ids) or best_move(i, all_ids)
            depth, tid = mv
            passes.append((tid, band, depth))
            current[i] -= depth
            last_tool[0] = tid

    out = [str(len(passes))]
    for (t, b, d) in passes:
        out.append(f"{t} {b} {d}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

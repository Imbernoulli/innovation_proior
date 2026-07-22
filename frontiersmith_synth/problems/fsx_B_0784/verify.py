#!/usr/bin/env python3
# Deterministic checker for "Gantry Heads on a Mural Rail" (format C, minimize makespan).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys

MAXOPS_PER_HEAD = 6000
MAXTICK = 400000


class Infeasible(Exception):
    pass


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def simulate(H, L, D, S, starts, strokes_by_pos, N, ops_per_head):
    """Tick-accurate replay of the gantry rail. ops_per_head[h] is a list of
    ('M', dx) | ('S', c) | ('P', None). Returns (finish_time list, painted set).
    Raises Infeasible on any rule violation."""
    pos = list(starts)
    color = [0] * H            # 0 = no color loaded
    ptr = [0] * H
    finished = [False] * H
    finish_time = [0] * H
    busy_until = [None] * H
    busy_type = [None] * H
    busy_target = [None] * H
    painted = set()

    def try_start(h, t):
        if ptr[h] >= len(ops_per_head[h]):
            finished[h] = True
            finish_time[h] = t
            busy_until[h] = None
            return
        kind, arg = ops_per_head[h][ptr[h]]
        if kind == 'M':
            newpos = pos[h] + arg
            if newpos < 0 or newpos > L:
                raise Infeasible("head %d move out of [0,%d]" % (h, L))
            throttled = False
            for h2 in range(H):
                if h2 != h and abs(pos[h2] - pos[h]) <= D:
                    throttled = True
                    break
            dur = 2 if throttled else 1
            busy_until[h] = t + dur
            busy_type[h] = 'M'
            busy_target[h] = newpos
        elif kind == 'S':
            busy_until[h] = t + S
            busy_type[h] = 'S'
            busy_target[h] = arg
        elif kind == 'P':
            busy_until[h] = t + 1
            busy_type[h] = 'P'
            busy_target[h] = None
        else:
            raise Infeasible("bad op kind")

    for h in range(H):
        try_start(h, 0)

    while not all(finished):
        active = [busy_until[h] for h in range(H) if not finished[h]]
        if not active:
            break
        t = min(active)
        if t > MAXTICK:
            raise Infeasible("exceeded tick budget")
        # complete everything due at t
        for h in range(H):
            if finished[h] or busy_until[h] != t:
                continue
            if busy_type[h] == 'M':
                pos[h] = busy_target[h]
            elif busy_type[h] == 'S':
                color[h] = busy_target[h]
            elif busy_type[h] == 'P':
                if pos[h] not in strokes_by_pos:
                    raise Infeasible("head %d paint at position %d with no stroke" % (h, pos[h]))
                req = strokes_by_pos[pos[h]]
                if color[h] != req:
                    raise Infeasible("head %d wrong color at %d (loaded %d need %d)" % (h, pos[h], color[h], req))
                if pos[h] in painted:
                    raise Infeasible("head %d double-paints position %d" % (h, pos[h]))
                painted.add(pos[h])
            ptr[h] += 1
        # start next ops for the heads that just advanced
        for h in range(H):
            if finished[h] or busy_until[h] != t:
                continue
            try_start(h, t)

    if len(painted) != N:
        raise Infeasible("only %d/%d strokes painted" % (len(painted), N))
    return finish_time


def parse_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    L = int(next(it)); H = int(next(it)); D = int(next(it)); S = int(next(it))
    K = int(next(it)); N = int(next(it))
    starts = [int(next(it)) for _ in range(H)]
    order = []  # (pos, color) in the given order
    strokes_by_pos = {}
    for _ in range(N):
        p = int(next(it)); c = int(next(it))
        order.append((p, c))
        strokes_by_pos[p] = c
    return L, H, D, S, K, N, starts, order, strokes_by_pos


def parse_output(path, H):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("no output")
    it = iter(toks)
    try:
        Hp = int(next(it))
    except (StopIteration, ValueError):
        fail("bad/empty output")
    if Hp != H:
        fail("head count mismatch: expected %d got %d" % (H, Hp))
    ops_per_head = []
    for h in range(H):
        try:
            m = int(next(it))
        except (StopIteration, ValueError):
            fail("missing op count for head %d" % h)
        if m < 0 or m > MAXOPS_PER_HEAD:
            fail("head %d op count %d out of [0,%d]" % (h, m, MAXOPS_PER_HEAD))
        ops = []
        for _ in range(m):
            try:
                tok = next(it)
            except StopIteration:
                fail("truncated op stream for head %d" % h)
            if tok == 'M':
                try:
                    dx = int(next(it))
                except (StopIteration, ValueError):
                    fail("bad move arg for head %d" % h)
                if dx not in (-1, 1):
                    fail("head %d move step must be +-1, got %d" % (h, dx))
                ops.append(('M', dx))
            elif tok == 'S':
                try:
                    c = int(next(it))
                except (StopIteration, ValueError):
                    fail("bad swap arg for head %d" % h)
                ops.append(('S', c))
            elif tok == 'P':
                ops.append(('P', None))
            else:
                fail("head %d unknown op token %r" % (h, tok))
        ops_per_head.append(ops)
    # no trailing garbage required to match exactly; extra tokens are ignored
    return ops_per_head


def build_baseline_ops(H, starts, order):
    """Baseline B: head 0 alone visits every stroke in the GIVEN input order,
    moving directly (unit steps) to each stroke's position, swapping whenever the
    needed color differs from what's currently loaded, then painting. Every other
    head never acts. No throttling ever occurs (only one head ever moves)."""
    ops = [[] for _ in range(H)]
    pos = starts[0]
    cur_color = 0
    for (p, c) in order:
        dx = 1 if p >= pos else -1
        while pos != p:
            ops[0].append(('M', dx))
            pos += dx
        if cur_color != c:
            ops[0].append(('S', c))
            cur_color = c
        ops[0].append(('P', None))
    return ops


def main():
    L, H, D, S, K, N, starts, order, strokes_by_pos = parse_instance(sys.argv[1])

    ops_per_head = parse_output(sys.argv[2], H)
    for h in range(H):
        if len(ops_per_head[h]) > MAXOPS_PER_HEAD:
            fail("head %d exceeds op budget" % h)

    try:
        finish_time = simulate(H, L, D, S, starts, strokes_by_pos, N, ops_per_head)
    except Infeasible as e:
        fail(str(e))

    F = max(finish_time)
    if F <= 0:
        fail("degenerate zero makespan")

    base_ops = build_baseline_ops(H, starts, order)
    base_finish = simulate(H, L, D, S, starts, strokes_by_pos, N, base_ops)
    B = max(base_finish)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

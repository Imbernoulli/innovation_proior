#!/usr/bin/env python3
"""
Trivial baseline for ale-11: a myopic greedy dispatcher.

Reads the instance on stdin, writes a (feasible) solution on stdout. Used only
to establish the bar the Hungarian solver must beat.

Policy: process requests in release-time order. For each request, among the
vehicles that can currently serve it, give it to the one that can pick it up
earliest (nearest in completion time); skip the request if no vehicle can.
This is the "nearest free vehicle grabs the request" greedy -- exactly the
myopic approach the rolling-horizon Hungarian is meant to beat.
"""
import sys


def man(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V = int(next(it)); N = int(next(it)); T = int(next(it))
    vx = []; vy = []
    for _ in range(V):
        vx.append(int(next(it))); vy.append(int(next(it)))
    req = []
    for _ in range(N):
        px = int(next(it)); py = int(next(it))
        qx = int(next(it)); qy = int(next(it))
        r = int(next(it)); e = int(next(it))
        req.append((px, py, qx, qy, r, e))

    cx = list(vx); cy = list(vy); ft = [0] * V

    order = sorted(range(N), key=lambda i: req[i][4])  # by release
    out = []
    for i in order:
        px, py, qx, qy, r, e = req[i]
        ride = man(px, py, qx, qy)
        best_v = -1; best_finish = None
        for v in range(V):
            st = ft[v] + man(cx[v], cy[v], px, py)
            if st < r:
                st = r
            if st > e:
                continue
            finish = st + ride
            if finish > T:
                continue
            if best_finish is None or finish < best_finish:
                best_finish = finish; best_v = v
        if best_v >= 0:
            v = best_v
            cx[v] = qx; cy[v] = qy; ft[v] = best_finish
            out.append((v, i))

    lines = [str(len(out))]
    for (v, i) in out:
        lines.append(f"{v} {i}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

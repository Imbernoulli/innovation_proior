# TIER: trivial
# Serial one-truck baseline: truck 0 alone does repeated complete round trips (with its
# own just-in-time cooldown stops), while every other truck just sits at the pit forever.
# This reproduces the checker's internal baseline B exactly, so it scores ~0.1.
import sys


def descent_checkpoints(t_list, g_list, H_MAX, idle_cool, t0):
    cps = []
    heat = 0
    t_cur = t0
    for i in range(len(t_list)):
        g = g_list[i]
        if heat + g > H_MAX:
            need = heat + g - H_MAX
            w = -(-need // idle_cool)
            if w > 0:
                heat = max(0, heat - w * idle_cool)
                t_cur += w
                cps.append((t_cur, i))
        t_cur += t_list[i]
        heat += g
        cps.append((t_cur, i + 1))
    return cps, t_cur


def ascent_checkpoints(t_list, t0, M):
    cps = []
    t_cur = t0
    for i in reversed(range(len(t_list))):
        t_cur += t_list[i]
        cps.append((t_cur, i))
    return cps, t_cur


def main():
    data = sys.stdin.read().split()
    p = iter(data)
    M = int(next(p)); K = int(next(p)); T = int(next(p))
    H_MAX = int(next(p)); idle_cool = int(next(p)); heat_loss = int(next(p))
    t_list = [int(next(p)) for _ in range(M)]
    g_list = [int(next(p)) for _ in range(M)]

    cps = [(0, 0)]
    t_cur = 0
    while True:
        seg, t_cur2 = descent_checkpoints(t_list, g_list, H_MAX, idle_cool, t_cur)
        if not seg or seg[-1][0] > T:
            break
        cps.extend(seg)
        t_cur = t_cur2
        seg, t_cur2 = ascent_checkpoints(t_list, t_cur, M)
        if not seg or seg[-1][0] > T:
            break
        cps.extend(seg)
        t_cur = t_cur2

    out = [str(K)]
    line = [str(len(cps))]
    for (tk, nd) in cps:
        line.append(str(tk)); line.append(str(nd))
    out.append(" ".join(line))
    for _ in range(K - 1):
        out.append("1 0 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

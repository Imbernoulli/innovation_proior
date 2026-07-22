# TIER: invalid
# Every truck follows the identical fastest schedule, so all K trucks pile into block 1
# (and every later block, and every pullout) at the exact same ticks -> block/pullout
# capacity violated immediately -> Ratio: 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)
    M = int(next(p)); K = int(next(p)); T = int(next(p))
    H_MAX = int(next(p)); idle_cool = int(next(p)); heat_loss = int(next(p))
    t_list = [int(next(p)) for _ in range(M)]
    g_list = [int(next(p)) for _ in range(M)]

    cps = [(0, 0)]
    t_cur = 0
    for i in range(M):
        t_cur += t_list[i]
        cps.append((t_cur, i + 1))
        if t_cur > T:
            break

    out = [str(K)]
    line = [str(len(cps))]
    for (tk, nd) in cps:
        line.append(str(tk)); line.append(str(nd))
    row = " ".join(line)
    for _ in range(K):
        out.append(row)  # every truck uses the identical timeline -> mass collision
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

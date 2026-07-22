# TIER: trivial
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

    # any tool with minDepth==1 (generator always provides one) -- mirrors the checker baseline
    fin_tool = None
    for idx, (mx, mn) in enumerate(tools):
        if mn == 1:
            fin_tool = idx + 1
            break
    if fin_tool is None:
        fin_tool = 1
    mx_fin, mn_fin = tools[fin_tool - 1]

    order = sorted(range(n), key=lambda i: (-target[i], i))

    current = [R] * n
    passes = []
    for i in order:
        band = i + 1
        while current[i] > target[i]:
            headroom = current[i] - target[i]
            S = min(current[0:band])
            cap_chatter = (K * S) // band
            depth = min(mx_fin, headroom, cap_chatter)
            if depth < 1:
                depth = 1
            passes.append((fin_tool, band, depth))
            current[i] -= depth

    out = [str(len(passes))]
    for (t, b, d) in passes:
        out.append(f"{t} {b} {d}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

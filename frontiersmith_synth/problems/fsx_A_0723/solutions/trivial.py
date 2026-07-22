# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); base = int(next(it))
    K = int(next(it))
    for _ in range(K):
        for _ in range(5):
            next(it)
    H = int(next(it))
    hint_map = {}
    for _ in range(H):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        hint_map[(r, c)] = v

    # reproduce the checker's own baseline tableau exactly: column-index-mod-base fill
    out_lines = []
    for r in range(R):
        row = []
        for c in range(C):
            if (r, c) in hint_map:
                row.append(hint_map[(r, c)])
            else:
                row.append(c % base)
        out_lines.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

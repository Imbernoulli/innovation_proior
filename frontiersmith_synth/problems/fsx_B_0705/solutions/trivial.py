# TIER: trivial
"""
Reproduces the checker's own baseline exactly: spend all raw material R on
group 0's direct route only. Never touches Y, Z, or any Combo reaction, and
never even looks at any group other than group 0.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    N_R = int(data[idx]); idx += 1
    N_Z = int(data[idx]); idx += 1
    groups = []
    for g in range(m):
        vals = [int(data[idx + k]) for k in range(7)]
        idx += 7
        groups.append(vals)

    px0, py0, dx0, dv0, cx0, cy0, cw0 = groups[0]
    denom = dx0 * px0
    f = N_R // denom if denom > 0 else 0

    tokens = []
    tokens.extend(["0X"] * (f * dx0))
    tokens.extend(["0D"] * f)

    out = []
    out.append(str(len(tokens)))
    out.append(" ".join(tokens))
    print("\n".join(out))


if __name__ == "__main__":
    main()

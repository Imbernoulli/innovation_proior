# TIER: trivial
import sys


def band_centers(r_in, r_out, Q):
    span = r_out - r_in
    seg = span // Q
    centers = []
    for i in range(Q):
        lo = r_in + i * seg
        hi = r_in + (i + 1) * seg if i < Q - 1 else r_out
        centers.append((lo + hi) // 2)
    return centers


def main():
    r_in, r_out, K, Q, S, M = (int(x) for x in sys.stdin.read().split()[:6])
    centers = band_centers(r_in, r_out, Q)
    lines = [str(len(centers))]
    for C in centers:
        r, d, p = 2, 1, 0
        R = C + r
        lines.append(f"{R} {r} {d} {p}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

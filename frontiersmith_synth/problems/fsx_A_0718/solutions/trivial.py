# TIER: trivial
import sys

def main():
    data = sys.stdin.read().split()
    pos = 0
    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v
    S = int(nxt()); K = int(nxt()); P = int(nxt()); R = int(nxt()); T = int(nxt())
    sats = []
    for _ in range(S):
        acc = int(nxt()); cap = int(nxt()); ph = int(nxt())
        sats.append((acc, cap, ph))
    stations = []
    for _ in range(K):
        drain = int(nxt()); off = int(nxt()); dur = int(nxt())
        stations.append((drain, off, dur))

    drain0, off0, dur0 = stations[0]
    out = []
    cnt = 0
    for i in range(S):
        ph = sats[i][2]
        base = ph + off0
        for c in range(R):
            s = c * P + base
            e = s + dur0
            out.append(f"{i} 0 {s} {e}")
            cnt += 1
    print(cnt)
    if out:
        sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()

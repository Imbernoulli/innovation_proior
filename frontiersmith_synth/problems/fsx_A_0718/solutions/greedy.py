# TIER: greedy
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

    # obvious approach: whenever a satellite is visible to a station, transmit --
    # grab every opportunity, in isolation, without checking who else is on the
    # same station at the same time.
    out = []
    cnt = 0
    for i in range(S):
        ph = sats[i][2]
        for k in range(K):
            drain, off, dur = stations[k]
            base = ph + off
            for c in range(R):
                s = c * P + base
                e = s + dur
                out.append(f"{i} {k} {s} {e}")
                cnt += 1
    print(cnt)
    if out:
        sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()

# TIER: greedy
# The textbook priority-list unit commitment: pick, once and for all, the cheapest
# prefix of boilers whose combined capacity covers PEAK demand, keep exactly those
# online for the whole horizon, and split each step's demand among them in proportion
# to capacity.  It saves the no-load fuel of the boilers it never lights -- but it
# commits a static fleet sized for the peak and lets it run cold at the shoulders,
# and it never steers loads onto the efficiency sweet spot.
import sys


def main():
    toks = sys.stdin.read().split()
    i = 0
    T = int(toks[i]); i += 1
    K = int(toks[i]); i += 1
    D = [float(toks[i + j]) for j in range(T)]; i += T
    caps = []; pmins = []; cs = []; as_ = []; bs = []; xs = []
    for _ in range(K):
        caps.append(float(toks[i])); pmins.append(float(toks[i + 1]))
        cs.append(float(toks[i + 2])); as_.append(float(toks[i + 3]))
        bs.append(float(toks[i + 4])); xs.append(float(toks[i + 5])); i += 9

    def avgcost(k):
        full = cs[k] + as_[k] * caps[k] * (1.0 + bs[k] * (1.0 - xs[k]) ** 2)
        return full / caps[k]

    order = sorted(range(K), key=avgcost)
    dmax = max(D)
    S = []; acc = 0.0
    for k in order:
        S.append(k); acc += caps[k]
        if acc >= dmax:
            break
    Scap = sum(caps[k] for k in S)

    out = []
    for t in range(T):
        row = [0.0] * K
        for k in S:
            share = D[t] * caps[k] / Scap
            row[k] = share if share > pmins[k] else pmins[k]
        out.append(" ".join("%.6f" % v for v in row))
    sys.stdout.write("\n".join(out) + "\n")


main()

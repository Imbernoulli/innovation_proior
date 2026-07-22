import sys
from fractions import Fraction as Fr


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = toks[p]; p += 1
        return v
    N = int(nxt()); T = int(nxt()); K = int(nxt())
    alpha = Fr(int(nxt()), 1000)
    decay = Fr(int(nxt()), 1000)
    gamma = Fr(int(nxt()), 1000)
    songs = []
    for _ in range(N):
        e = int(nxt()); d = int(nxt())
        style = [int(nxt()) for _ in range(K)]
        songs.append((e, d, style))
    return N, T, K, alpha, decay, gamma, songs


def simulate(order, songs, K, alpha, decay, gamma):
    """Exact (Fraction) replay of the crowd dynamics. Returns raw score in [0,1]."""
    E = Fr(0)
    M = [Fr(0)] * K
    peak = Fr(0)
    for idx in order:
        e_milli, d, style = songs[idx]
        e = Fr(e_milli, 1000)
        sim = sum(Fr(style[j]) * M[j] for j in range(K)) / K
        fatigue = gamma * sim * E
        E = E + e * (1 - E) - fatigue
        if E < 0:
            E = Fr(0)
        if E > 1:
            E = Fr(1)
        if E > peak:
            peak = E
        for j in range(K):
            M[j] = decay * M[j] + (1 - decay) * Fr(style[j])
    final = E
    return alpha * peak + (1 - alpha) * final


def internal_baseline(songs, T):
    """Checker's own trivial, unambitious construction: the calmest songs first
    (ascending energy, ties by duration then index), and don't even fill the whole
    time budget -- stop once 2/5 of T is used. A deliberately unstrategic reference."""
    cap = (2 * T) // 5
    idxs = sorted(range(len(songs)), key=lambda i: (songs[i][0], songs[i][1], i))
    order = []
    tot = 0
    for i in idxs:
        d = songs[i][1]
        if tot + d <= cap:
            order.append(i)
            tot += d
    return order


def read_participant(path, N):
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        return None, "empty output"
    try:
        m = int(toks[0])
    except ValueError:
        return None, "non-integer count"
    if m < 0 or m > N:
        return None, "count out of range"
    if len(toks) < 1 + m:
        return None, "not enough index tokens"
    order = []
    for t in toks[1:1 + m]:
        try:
            v = int(t)
        except ValueError:
            return None, "non-integer index"
        order.append(v)
    # extra trailing tokens beyond the declared m are ignored (schema is m then m indices)
    return order, None


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad args)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, T, K, alpha, decay, gamma, songs = read_instance(in_path)

    order, err = read_participant(out_path, N)
    if err is not None:
        print(f"Ratio: 0.0 ({err})")
        return 0

    # feasibility: distinct, in-range indices; must be finite integers (guaranteed by int()
    # parsing above -- nan/inf tokens fail int() and are already rejected as non-integer).
    if len(set(order)) != len(order):
        print("Ratio: 0.0 (duplicate index)")
        return 0
    for v in order:
        if v < 0 or v >= N:
            print("Ratio: 0.0 (index out of range)")
            return 0
    total_dur = sum(songs[v][1] for v in order)
    if total_dur > T:
        print("Ratio: 0.0 (over time budget)")
        return 0

    F = simulate(order, songs, K, alpha, decay, gamma)
    if F < 0 or F > 1:
        print("Ratio: 0.0 (score out of range)")
        return 0

    base_order = internal_baseline(songs, T)
    B = simulate(base_order, songs, K, alpha, decay, gamma)
    if B <= 0:
        print("Ratio: 0.0 (degenerate baseline)")
        return 0

    sc = min(Fr(1000), Fr(100) * F / B)
    ratio = float(sc) / 1000.0
    print("Ratio: %.6f" % ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())

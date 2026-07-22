import sys, math

EPS = 0.05  # small positive floor so a length-1 melody has F > 0


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    K = int(nxt()); L = int(nxt())
    C = [[0] * 12 for _ in range(12)]
    for i in range(12):
        for j in range(12):
            C[i][j] = int(nxt())
    voices = []
    for _ in range(K):
        d = int(nxt()); t = int(nxt())
        voices.append((d, t))
    return K, L, C, voices


def derive_rules(voices):
    """Every simultaneity constraint created by overlaying the original melody
    with all delayed/transposed copies of itself reduces to a family of rules
    of the form: for all valid a, pitch[a+Delta] must be consonant (via C)
    with (pitch[a] + delta_t) mod 12.  Delta=d_i,delta_t=t_i comes from the
    original vs copy i; Delta=|d_i-d_j|, delta_t = (t_lo - t_hi) mod 12
    (lo = the voice with the smaller delay) comes from copy i vs copy j
    overlapping each other."""
    rules = []
    for (d, t) in voices:
        rules.append((d, t % 12))
    n = len(voices)
    for i in range(n):
        for j in range(i + 1, n):
            d1, t1 = voices[i]
            d2, t2 = voices[j]
            if d1 == d2:
                continue
            if d1 < d2:
                Delta = d2 - d1
                delta_t = (t1 - t2) % 12
            else:
                Delta = d1 - d2
                delta_t = (t2 - t1) % 12
            rules.append((Delta, delta_t))
    return rules


def check_feasible(c, C, rules):
    m = len(c)
    for (Delta, delta_t) in rules:
        if Delta >= m:
            continue
        for a in range(0, m - Delta):
            x = c[a + Delta]
            y = (c[a] + delta_t) % 12
            if C[x][y] != 1:
                return False
    return True


def score_melody(c):
    m = len(c)
    cnt = [0] * 12
    for p in c:
        cnt[p] += 1
    H = 0.0
    for k in cnt:
        if k:
            pr = k / m
            H -= pr * math.log2(pr)
    H_norm = H / math.log2(12)

    TG = 0
    if m >= 4:
        steps = []
        for i in range(m - 1):
            diff = ((c[i + 1] - c[i] + 6) % 12) - 6
            steps.append(1 if diff > 0 else (-1 if diff < 0 else 0))
        seen = set()
        for i in range(len(steps) - 2):
            seen.add((steps[i], steps[i + 1], steps[i + 2]))
        TG = len(seen)

    F = m * (H_norm + EPS) + TG
    return F


def baseline(L, C, rules):
    """The trivial recipe: try every 2-note alternating pattern p,q,p,q,...
    (a drone is the p==q special case), keep the longest feasible prefix of
    each, and take the single best one found. No lookahead, no adaptation
    to the developing melody -- just an exhaustive scan of a fixed, tiny
    family of periodic patterns."""
    best_c = [0]
    best_F = -1.0
    for p in range(12):
        for q in range(12):
            c = []
            m = 0
            for i in range(L):
                c.append(p if i % 2 == 0 else q)
                if not check_feasible(c, C, rules):
                    c.pop()
                    break
                m = i + 1
            if m == 0:
                continue
            F = score_melody(c)
            if F > best_F:
                best_F = F
                best_c = c
    return score_melody(best_c)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    K, L, C, voices = read_instance(in_path)
    rules = derive_rules(voices)

    with open(out_path) as f:
        toks = f.read().split()

    if len(toks) < 1:
        print("empty output Ratio: 0.0")
        return 0
    try:
        m = int(toks[0])
    except ValueError:
        print("bad length token Ratio: 0.0")
        return 0
    if m < 1 or m > L:
        print("length out of range Ratio: 0.0")
        return 0
    if len(toks) != 1 + m:
        print("token count mismatch Ratio: 0.0")
        return 0
    c = []
    for i in range(m):
        try:
            v = int(toks[1 + i])
        except ValueError:
            print("bad pitch token Ratio: 0.0")
            return 0
        if not (0 <= v <= 11):
            print("pitch out of range Ratio: 0.0")
            return 0
        c.append(v)

    if not check_feasible(c, C, rules):
        print("self-overlap consonance violated Ratio: 0.0")
        return 0

    F = score_melody(c)
    B = baseline(L, C, rules)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

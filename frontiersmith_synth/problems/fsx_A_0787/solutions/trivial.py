# TIER: trivial
# Scan the fixed, tiny family of 2-note alternating patterns p,q,p,q,...
# (a drone is the p==q special case) and keep the longest-feasible, best
# scoring one. No adaptation to the developing melody, no lookahead beyond
# "does extending this fixed pattern by one more note still work" -- this is
# exactly the checker's own internal baseline construction.
import sys


def read_instance():
    toks = sys.stdin.read().split()
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
    import math
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
    return m * (H_norm + 0.05) + TG


def main():
    K, L, C, voices = read_instance()
    rules = derive_rules(voices)
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
    print(len(best_c))
    print(" ".join(map(str, best_c)))


if __name__ == "__main__":
    main()

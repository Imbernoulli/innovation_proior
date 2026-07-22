# TIER: greedy
# The obvious first attempt: extend the melody one note at a time,
# left-to-right. To keep pitch usage roughly balanced (an "obvious" nod
# towards higher entropy) it tries the currently least-used pitch class
# first, falling back through the rest in a fixed order, taking the first
# one that is consonant with everything already written (all
# delayed/transposed echoes whose constraint against the new note is
# already checkable). What it never does is look ahead: it never asks
# whether today's "safe, balanced" choice will trap tomorrow's note between
# several incompatible echoes. On sparsely-consonant instances this
# deadlocks early and the melody truncates -- no backtracking, no recovery.
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


def valid_next(c, s, cand, C, rules):
    for (Delta, delta_t) in rules:
        if Delta > s:
            continue
        y = (c[s - Delta] + delta_t) % 12
        if C[cand][y] != 1:
            return False
    return True


def main():
    K, L, C, voices = read_instance()
    rules = derive_rules(voices)
    c = []
    usage = [0] * 12
    for s in range(L):
        placed = False
        order = sorted(range(12), key=lambda p: (usage[p], p))
        for cand in order:
            if valid_next(c, s, cand, C, rules):
                c.append(cand)
                usage[cand] += 1
                placed = True
                break
        if not placed:
            break
    if not c:
        c = [0]
    print(len(c))
    print(" ".join(map(str, c)))


if __name__ == "__main__":
    main()

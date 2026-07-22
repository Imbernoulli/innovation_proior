# TIER: greedy
# The obvious first attempt: to maximize motif diversity, just fill the threading and treadling
# with a varied (pseudo-random) assignment and repair any illegal floats. It reads plausible and
# does raise window diversity -- but it treats the drawdown as a free pixel canvas and FORGETS the
# other half of the score: symmetry. Random factors give mirror/rot agreement ~1/2, so the whole
# design is multiplied by ~0.5. The insight it misses (see strong): symmetry is FREE if the
# threading/treadling factors are made palindromic.
import sys

def make_tieup(S, T):
    def lcg(x):
        return (x * 1103515245 + 12345) & 0x7fffffff
    x = 2654435761 ^ (S * 131 + T * 977)
    for _ in range(4000):
        x = lcg(x)
        M = [[0] * S for _ in range(T)]
        r = x
        for t in range(T):
            for s in range(S):
                r = lcg(r)
                M[t][s] = (r >> 16) & 1
        if any(all(M[t][s] == M[t][0] for s in range(S)) for t in range(T)):
            continue
        cols = [tuple(M[t][s] for t in range(T)) for s in range(S)]
        if any(all(v == col[0] for v in col) for col in cols):
            continue
        if len(set(cols)) != S or len(set(tuple(M[t]) for t in range(T))) != T:
            continue
        return M
    return [[1 if ((s + 2 * t) % S) < (S // 2) else 0 for s in range(S)] for t in range(T)]

def excess(seq, channel_val, num_ch, L):
    tot = 0
    n = len(seq)
    for c in range(num_ch):
        prev = channel_val(c, seq[0]); run = 1
        for j in range(1, n):
            v = channel_val(c, seq[j])
            if v == prev:
                run += 1
            else:
                if run > L:
                    tot += run - L
                run = 1; prev = v
        if run > L:
            tot += run - L
    return tot

def first_violation(seq, channel_val, num_ch, L):
    n = len(seq)
    for c in range(num_ch):
        prev = channel_val(c, seq[0]); run = 1; start = 0
        for j in range(1, n):
            v = channel_val(c, seq[j])
            if v == prev:
                run += 1
            else:
                if run > L:
                    return start + run // 2
                run = 1; prev = v; start = j
        if run > L:
            return start + run // 2
    return -1

def repair(seq, A, channel_val, num_ch, L):
    it = 0
    while it < 1000:
        it += 1
        p = first_violation(seq, channel_val, num_ch, L)
        if p < 0:
            break
        orig = seq[p]; best = orig; bestval = excess(seq, channel_val, num_ch, L)
        for s in range(1, A + 1):
            if s == orig:
                continue
            seq[p] = s
            val = excess(seq, channel_val, num_ch, L)
            if val < bestval:
                bestval = val; best = s
        seq[p] = best
        if best == orig:
            seq[p] = (orig % A) + 1
    return seq

def lcg(x):
    return (x * 1103515245 + 12345) & 0x7fffffff

def build_full_random(N, A, channel_val, num_ch, L, seed):
    # naive-but-careful: at each position pick a RANDOM float-safe symbol (no symmetry planning)
    run_val = [-1] * num_ch
    run_len = [0] * num_ch
    seq = []
    r = seed & 0x7fffffff
    for _ in range(N):
        safe = []
        for s in range(1, A + 1):
            ok = True
            for c in range(num_ch):
                v = channel_val(c, s)
                nl = run_len[c] + 1 if v == run_val[c] else 1
                if nl > L:
                    ok = False; break
            if ok:
                safe.append(s)
        if not safe:
            best = 1; bestmax = 1 << 30
            for s in range(1, A + 1):
                mx = 0
                for c in range(num_ch):
                    v = channel_val(c, s)
                    nl = run_len[c] + 1 if v == run_val[c] else 1
                    if nl > mx:
                        mx = nl
                if mx < bestmax:
                    bestmax = mx; best = s
            safe = [best]
        r = lcg(r)
        chosen = safe[(r >> 8) % len(safe)]
        seq.append(chosen)
        for c in range(num_ch):
            v = channel_val(c, chosen)
            if v == run_val[c]:
                run_len[c] += 1
            else:
                run_val[c] = v; run_len[c] = 1
    return seq

def main():
    tok = sys.stdin.read().split()
    N = int(tok[0]); S = int(tok[1]); T = int(tok[2]); L = int(tok[3])
    tieup = make_tieup(S, T)
    th_cv = lambda c, s: tieup[c][s - 1]
    tr_cv = lambda c, t: tieup[t - 1][c]
    threading = build_full_random(N, S, th_cv, T, L, 0xC0FFEE ^ (N * 3 + S * 5))
    treadling = build_full_random(N, T, tr_cv, S, L, 0xBADA55 ^ (N * 7 + T * 11))
    repair(threading, S, th_cv, T, L)
    repair(treadling, T, tr_cv, S, L)
    out = []
    out.append(" ".join(map(str, threading)))
    out.append(" ".join(map(str, treadling)))
    for t in range(T):
        out.append(" ".join(map(str, tieup[t])))
    sys.stdout.write("\n".join(out) + "\n")

main()

# TIER: strong
# INSIGHT (the innovation hook): you cannot edit the drawdown pixel by pixel -- it lives in
# the rank-<=S boolean product tieup[treadling[i]][threading[j]]. Every legal move is a change
# to a threading/treadling symbol, which rewrites a whole column/row class at once. So:
#   (1) get all symmetry for free by making the FACTORS palindromic (threading[j]=threading[N-1-j],
#       treadling[i]=treadling[N-1-i])  ->  mirror + rot180 agreement = 1 exactly.
#   (2) the float-cap DECOUPLES: a horizontal float depends only on (threading, tieup) per treadle;
#       a vertical float only on (treadling, tieup) per shaft. So enforce runs channel-by-channel
#       on the two 1-D factor sequences instead of on the 2-D artifact.
#   (3) within those constraints, treat each half-sequence as a de-Bruijn-like code that maximizes
#       the number of DISTINCT length-k symbol windows -> maximal motif diversity, not a monotone scan.
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
        # rows non-constant, columns non-constant, all columns distinct, all rows distinct
        if any(all(M[t][s] == M[t][0] for s in range(S)) for t in range(T)):
            continue
        cols = [tuple(M[t][s] for t in range(T)) for s in range(S)]
        if any(all(v == col[0] for v in col) for col in cols):
            continue
        if len(set(cols)) != S:
            continue
        rows = [tuple(M[t]) for t in range(T)]
        if len(set(rows)) != T:
            continue
        return M
    # deterministic fallback: broken twill
    return [[1 if ((s + 2 * t) % S) < (S // 2) else 0 for s in range(S)] for t in range(T)]

def build_half(hc, A, channel_val, num_ch, L, k, seed):
    run_val = [-1] * num_ch
    run_len = [0] * num_ch
    half = []
    seen = set()
    rng = seed & 0x7fffffff
    def rnd():
        nonlocal rng
        rng = (rng * 1103515245 + 12345) & 0x7fffffff
        return rng
    for q in range(hc):
        final = (q == hc - 1)
        cap = (L + 1) // 2 if final else L
        cands = []
        for s in range(1, A + 1):
            ok = True
            for c in range(num_ch):
                v = channel_val(c, s)
                nl = run_len[c] + 1 if v == run_val[c] else 1
                if nl > cap:
                    ok = False
                    break
            if ok:
                cands.append(s)
        if not cands:
            best = 1
            bestmax = 1 << 30
            for s in range(1, A + 1):
                mx = 0
                for c in range(num_ch):
                    v = channel_val(c, s)
                    nl = run_len[c] + 1 if v == run_val[c] else 1
                    if nl > mx:
                        mx = nl
                if mx < bestmax:
                    bestmax = mx
                    best = s
            cands = [best]
        order = sorted(cands, key=lambda s: (rnd(), s))
        chosen = None
        for s in order:
            if k > 1 and len(half) >= k - 1:
                win = tuple(half[len(half) - (k - 1):] + [s])
            elif k == 1:
                win = (s,)
            else:
                win = None
            if win is not None and win not in seen:
                chosen = s
                break
        if chosen is None:
            chosen = order[0]
        half.append(chosen)
        if len(half) >= k:
            seen.add(tuple(half[len(half) - k:]))
        for c in range(num_ch):
            v = channel_val(c, chosen)
            if v == run_val[c]:
                run_len[c] += 1
            else:
                run_val[c] = v
                run_len[c] = 1
    return half

def palindrome(half, N):
    seq = [0] * N
    for j in range(len(half)):
        seq[j] = half[j]
        seq[N - 1 - j] = half[j]
    return seq

def _excess(half, N, channel_val, num_ch, L):
    full = palindrome(half, N)
    tot = 0
    for c in range(num_ch):
        prev = channel_val(c, full[0]); run = 1
        for j in range(1, N):
            v = channel_val(c, full[j])
            if v == prev:
                run += 1
            else:
                if run > L:
                    tot += run - L
                run = 1; prev = v
        if run > L:
            tot += run - L
    return tot

def _first_violation(half, N, channel_val, num_ch, L):
    full = palindrome(half, N)
    for c in range(num_ch):
        prev = channel_val(c, full[0]); run = 1; start = 0
        for j in range(1, N):
            v = channel_val(c, full[j])
            if v == prev:
                run += 1
            else:
                if run > L:
                    return start + run // 2
                run = 1; prev = v; start = j
        if run > L:
            return start + run // 2
    return -1

def repair(half, N, A, channel_val, num_ch, L):
    # A legal edit rewrites a whole row/col class: change one half-symbol (and its mirror)
    # to break the longest illegal float. Iterate until every channel's run <= L.
    it = 0
    while it < 500:
        it += 1
        p = _first_violation(half, N, channel_val, num_ch, L)
        if p < 0:
            break
        qh = min(p, N - 1 - p)
        orig = half[qh]
        best = orig
        bestval = _excess(half, N, channel_val, num_ch, L)
        for s in range(1, A + 1):
            if s == orig:
                continue
            half[qh] = s
            val = _excess(half, N, channel_val, num_ch, L)
            if val < bestval:
                bestval = val; best = s
        half[qh] = best
        if best == orig:                       # no strict improvement: nudge to avoid a stall
            half[qh] = (orig % A) + 1
    return half

def main():
    tok = sys.stdin.read().split()
    N = int(tok[0]); S = int(tok[1]); T = int(tok[2]); L = int(tok[3]); k = int(tok[4])
    tieup = make_tieup(S, T)
    hc = (N + 1) // 2

    # threading channels = treadles: horizontal float uses tieup[t][s-1]
    th_cv = lambda c, s: tieup[c][s - 1]
    th_half = build_half(hc, S, th_cv, T, L, k, 0x1234 ^ (N * 7 + S))
    th_half = repair(th_half, N, S, th_cv, T, L)
    # treadling channels = shafts: vertical float uses tieup[t-1][c]
    tr_cv = lambda c, t: tieup[t - 1][c]
    tr_half = build_half(hc, T, tr_cv, S, L, k, 0x9abc ^ (N * 13 + T))
    tr_half = repair(tr_half, N, T, tr_cv, S, L)

    threading = palindrome(th_half, N)
    treadling = palindrome(tr_half, N)

    out = []
    out.append(" ".join(map(str, threading)))
    out.append(" ".join(map(str, treadling)))
    for t in range(T):
        out.append(" ".join(map(str, tieup[t])))
    sys.stdout.write("\n".join(out) + "\n")

main()

#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for fsx_A_1364.

Feasibility: the batch I must respect BOTH partition-matroid quota systems.
Objective:   F = |I|, times a 1.2 bonus if a submitted certificate (A subset)
             satisfies r1(A) + r2(complement of A) == |I| EXACTLY -- which, by
             matroid-intersection weak duality, is only possible when |I| is
             truly maximum (so the bonus can never be earned by a suboptimal I).
Baseline B: the checker's own "Scheme-A-first, ignore Scheme-B, then repair"
            construction (always feasible, always positive).
"""
import sys


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 # bad invocation")
        return
    inpath, outpath = sys.argv[1], sys.argv[2]

    with open(inpath) as f:
        itoks = f.read().split()
    p = 0

    def in_int():
        nonlocal p
        v = int(itoks[p])
        p += 1
        return v

    n = in_int()
    K1 = in_int()
    K2 = in_int()
    col1 = [0] * n
    col2 = [0] * n
    for i in range(n):
        col1[i] = in_int() - 1
        col2[i] = in_int() - 1
    cap1 = [in_int() for _ in range(K1)]
    cap2 = [in_int() for _ in range(K2)]

    # ---- checker's own baseline construction ----
    cnt1 = [0] * K1
    S = []
    for i in range(n):
        c = col1[i]
        if cnt1[c] < cap1[c]:
            cnt1[c] += 1
            S.append(i)
    cnt2 = [0] * K2
    Bset = []
    for i in S:
        c = col2[i]
        if cnt2[c] < cap2[c]:
            cnt2[c] += 1
            Bset.append(i)
    Bsize = len(Bset)
    if Bsize <= 0:
        Bsize = 1  # defensive; construction is always >=1 by problem constraints

    # ---- read participant output as a free-form token stream ----
    try:
        with open(outpath) as f:
            otoks = f.read().split()
    except Exception:
        print("Ratio: 0.0 # cannot read output")
        return

    q = 0

    def next_int():
        nonlocal q
        if q >= len(otoks):
            return None
        tok = otoks[q]
        q += 1
        try:
            return int(tok)
        except Exception:
            return None

    m = next_int()
    if m is None or m < 0 or m > n:
        print("Ratio: 0.0 # bad batch size")
        return

    Iidx = []
    seen = set()
    for _ in range(m):
        v = next_int()
        if v is None or v < 1 or v > n or v in seen:
            print("Ratio: 0.0 # bad or duplicate item id")
            return
        seen.add(v)
        Iidx.append(v - 1)

    c1cnt = [0] * K1
    c2cnt = [0] * K2
    for idx in Iidx:
        c1cnt[col1[idx]] += 1
        c2cnt[col2[idx]] += 1
    for c in range(K1):
        if c1cnt[c] > cap1[c]:
            print("Ratio: 0.0 # scheme-A quota violated")
            return
    for c in range(K2):
        if c2cnt[c] > cap2[c]:
            print("Ratio: 0.0 # scheme-B quota violated")
            return

    # ---- optional maximality certificate ----
    bonus = False
    flag = next_int()
    if flag == 1:
        k = next_int()
        if k is not None and 0 <= k <= n:
            Aset = set()
            good = True
            for _ in range(k):
                v = next_int()
                if v is None or v < 1 or v > n or v in Aset:
                    good = False
                    break
                Aset.add(v - 1)
            if good:
                a1cnt = [0] * K1
                for idx in Aset:
                    a1cnt[col1[idx]] += 1
                r1 = sum(min(cap1[c], a1cnt[c]) for c in range(K1))
                b2cnt = [0] * K2
                for idx in range(n):
                    if idx not in Aset:
                        b2cnt[col2[idx]] += 1
                r2 = sum(min(cap2[c], b2cnt[c]) for c in range(K2))
                if r1 + r2 == m:
                    bonus = True

    F = float(m) * 1.2 if bonus else float(m)

    sc = min(1000.0, 100.0 * F / max(1e-9, float(Bsize)))
    print("BaselineSize: %d ChosenSize: %d CertBonus: %s Ratio: %.6f"
          % (Bsize, m, bonus, sc / 1000.0))


if __name__ == "__main__":
    main()

import sys

SUBS = {"A": "C", "C": "G", "G": "T", "T": "A"}
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def revcomp(s):
    return "".join(COMP[c] for c in reversed(s))


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        L = int(next(it)); M = int(next(it)); K = int(next(it))
        P = int(next(it)); W = int(next(it)); D = int(next(it))
        ref = next(it)
        positions = [int(next(it)) for _ in range(M)]
        snp_count = int(next(it))
        snp_offsets = [int(next(it)) for _ in range(snp_count)]
        tlen = int(next(it)); tstart = int(next(it))
        n_mut = int(next(it))
        mutants_spec = []
        for _ in range(n_mut):
            cls = int(next(it)); sz = int(next(it))
            idxs = [int(next(it)) for _ in range(sz)]
            mutants_spec.append((cls, idxs))
    except Exception:
        fail("bad input")
        return

    if len(ref) != L or M <= 0 or P <= 0 or W <= 0 or P > W or n_mut <= 0:
        fail("bad input shape")
        return

    def build_mutant(cls, idxs):
        buf = bytearray(ref.encode())
        for li in idxs:
            if li < 0 or li >= M:
                continue
            p = positions[li]
            window = ref[p:p + W]
            if cls == 0:
                wl = list(window)
                for o in snp_offsets:
                    if 0 <= o < len(wl):
                        wl[o] = SUBS[wl[o]]
                neww = "".join(wl)
            elif cls == 1:
                neww = revcomp(window)
            elif cls == 2:
                unit = window[tstart:tstart + tlen] if tlen > 0 else window
                reps = W // max(1, tlen) + 2
                neww = (unit * reps)[:W] if unit else window
            else:
                neww = window
            buf[p:p + W] = neww.encode()
        return buf.decode()

    mutants = [build_mutant(cls, idxs) for cls, idxs in mutants_spec]

    # A probe's CONTENT is fixed by (pos, orient) alone, but where it may BIND is not the
    # same as where it was drawn from once a locus has been reverse-complemented -- so
    # detection searches every aligned offset inside the locus window, not just `pos`
    # itself. Each probe is anchored to (at most) one locus -- the one whose window fully
    # contains its source position -- since loci never overlap.
    span = W - P

    def anchor(probes):
        contents = [ref[pos:pos + P] if orient == "F" else revcomp(ref[pos:pos + P])
                    for pos, orient in probes]
        by_locus = [[] for _ in range(M)]
        for k, (pos, orient) in enumerate(probes):
            for i in range(M):
                pi = positions[i]
                if pi <= pos <= pi + span:
                    by_locus[i].append(k)
                    break
        return contents, by_locus

    def score_probes(probes):
        contents, by_locus = anchor(probes)
        worst = 1.0
        for mut in mutants:
            hit = 0
            for i in range(M):
                plist = by_locus[i]
                if not plist:
                    continue
                pi = positions[i]
                found = False
                for q in range(pi, pi + span + 1):
                    seg = mut[q:q + P]
                    for k in plist:
                        content = contents[k]
                        mism = sum(1 for a, b in zip(content, seg) if a != b)
                        if mism <= D:
                            found = True
                            break
                    if found:
                        break
                if found:
                    hit += 1
            frac = hit / M
            if frac < worst:
                worst = frac
        return worst

    # ---- checker's own trivial baseline: 1 forward probe (shift 0) on the first C=max(2,M//4)
    #      marker loci only; these are guaranteed (by the generator) never to be edited ----
    C = max(2, M // 4)
    C = min(C, M)
    base_probes = [(positions[i], "F") for i in range(C)]
    B = max(score_probes(base_probes), 1e-9)

    # ---- parse participant output ----
    try:
        out = open(sys.argv[2]).read().split()
        ot = iter(out)
        T = int(next(ot))
    except Exception:
        fail("bad output header")
        return
    if T < 1 or T > K:
        fail("probe count %d out of budget [1,%d]" % (T, K))
        return

    probes = []
    try:
        for _ in range(T):
            pos = int(next(ot))
            orient = next(ot)
            if orient not in ("F", "R"):
                fail("bad orient %r" % orient)
                return
            if pos < 0 or pos > L - P:
                fail("pos out of range %d" % pos)
                return
            probes.append((pos, orient))
    except StopIteration:
        fail("truncated output")
        return
    except Exception:
        fail("bad probe token")
        return

    F = score_probes(probes)
    sc = min(1000.0, 100.0 * F / B)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

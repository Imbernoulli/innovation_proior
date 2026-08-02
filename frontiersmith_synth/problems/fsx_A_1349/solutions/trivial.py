# TIER: trivial
import sys


def vmod(bits, M):
    v = 0
    for c in bits:
        v = (2 * v + (1 if c == '1' else 0)) % M
    return v


def find_candidate_modulus(samples, cap=24):
    for Mc in range(2, cap + 1):
        resmap = {}
        ok = True
        for bits, label in samples:
            r = vmod(bits, Mc)
            if r in resmap:
                if resmap[r] != label:
                    ok = False
                    break
            else:
                resmap[r] = label
        if ok:
            return Mc, resmap
    return None, None


def main():
    toks = sys.stdin.read().split()
    it = 0
    seed = int(toks[it]); it += 1
    step_bound = int(toks[it]); it += 1
    n = int(toks[it]); it += 1
    samples = []
    for _ in range(n):
        bits = toks[it]; it += 1
        label = int(toks[it]); it += 1
        samples.append((bits, label))

    Mc, resmap = find_candidate_modulus(samples, cap=24)
    if Mc is None:
        sys.stdout.write("1\n0 0 R\n")
        return
    h = [resmap.get(r, 0) for r in range(Mc)]

    # Correctly identify the residue law, but represent state as a
    # (residue, redundant counter) pair without checking whether the
    # counter is ever actually needed -- a "correct but unminimized"
    # construction, KAUX times larger than necessary.
    KAUX = 8
    S = Mc * KAUX
    lines = [str(S)]
    for r in range(Mc):
        for k in range(KAUX):
            newr0 = (2 * r) % Mc
            newr1 = (2 * r + 1) % Mc
            newk = (k + 1) % KAUX
            idx0 = newr0 * KAUX + newk
            idx1 = newr1 * KAUX + newk
            tb = 'A' if h[r] == 1 else 'R'
            lines.append("%d %d %s" % (idx0, idx1, tb))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

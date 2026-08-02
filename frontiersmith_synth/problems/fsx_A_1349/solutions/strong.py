# TIER: strong
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


def refine(Mc, h):
    """Moore-style partition refinement: merge residues that are currently
    behaviourally indistinguishable (same accept status and same future
    under both symbols), iterating to a fixed point.  This is the
    behaviour-equivalence step -- it is what lets a large but correct
    hypothesis collapse to the true minimal state count."""
    trans0 = [(2 * r) % Mc for r in range(Mc)]
    trans1 = [(2 * r + 1) % Mc for r in range(Mc)]
    part = [h[r] for r in range(Mc)]
    for _ in range(Mc + 1):
        idmap = {}
        newpart = [0] * Mc
        nextid = 0
        for r in range(Mc):
            key = (part[r], part[trans0[r]], part[trans1[r]])
            if key not in idmap:
                idmap[key] = nextid
                nextid += 1
            newpart[r] = idmap[key]
        if newpart == part:
            break
        part = newpart

    groups = sorted(set(part))
    gindex = {g: i for i, g in enumerate(groups)}
    S = len(groups)
    rep = [None] * S
    for r in range(Mc):
        g = gindex[part[r]]
        if rep[g] is None:
            rep[g] = r
    rt0 = [0] * S
    rt1 = [0] * S
    racc = [0] * S
    for g in range(S):
        r = rep[g]
        rt0[g] = gindex[part[trans0[r]]]
        rt1[g] = gindex[part[trans1[r]]]
        racc[g] = h[r]
    return S, rt0, rt1, racc


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

    S, rt0, rt1, racc = refine(Mc, h)

    lines = [str(S)]
    for g in range(S):
        tb = 'A' if racc[g] == 1 else 'R'
        lines.append("%d %d %s" % (rt0[g], rt1[g], tb))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

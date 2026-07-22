import os
import random
import sys

# gen.py <testId> -- prints ONE instance to stdout.
#
# Instance = a small NFA over {0,1} defining a predicate on n-bit strings.
# The emitted transition tables come from three deterministic families:
#   popcount_exact : "the number of 1-bits is exactly k" (counter automaton)
#   popcount_mod   : "the number of 1-bits is r mod m" (cyclic automaton)
#   random         : rejection-sampled random DFA (seeded by testId), sometimes
#                    with planted CLONE states (pairs of states with identical
#                    futures, so the NFA is genuinely nondeterministic yet its
#                    subset space is unchanged -- an automaton-state-sharing
#                    easter egg the strong solution's CSE absorbs for free)
#
# TRAP design: every case has a language whose prime-implicant DNF stays fat
# (random tables barely merge; exact/modular popcount tables never merge at
# all: equal-weight minterms are pairwise non-adjacent), so the two-level
# Quine-McCluskey recipe stalls at ~1/6 of baseline, while the automaton has
# few LIVE subsets per input position and the shared backward circuit costs
# ~3*n*S_eff gates. Cases 1-3 plant this trap with named structures; the
# random cases plant it with structureless dense tables. (Substring-matcher
# automata were evaluated and REJECTED for the ladder: their two-level DNF is
# so good that the recipe would beat the insight -- see dev notes.)
#
# CALIBRATION: this generator imports the actual reference builders
# (solutions/strong.py, solutions/greedy.py) and computes each candidate
# instance's exact baseline B, strong gate count F_s and greedy gate count
# F_g before accepting it, so the emitted ladder lands inside the per-test
# ratio bands below. All randomness flows from testId only.

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "solutions"))
import strong as _strong
import greedy as _greedy

# ---------------------------------------------------------------- constructors

def popcount_exact(n, k):
    dead = k + 1
    S = k + 2
    trans = {}
    for c in range(k + 1):
        trans[(c, 0)] = (c,)
        trans[(c, 1)] = (c + 1,)
    trans[(dead, 0)] = (dead,)
    trans[(dead, 1)] = (dead,)
    return n, S, 0, [k], trans


def popcount_mod(n, m, r):
    trans = {}
    for c in range(m):
        trans[(c, 0)] = (c,)
        trans[(c, 1)] = ((c + 1) % m,)
    return n, m, 0, [r], trans


def _make_matcher(word):
    bits = [int(c) for c in word]
    k = len(bits)
    pi = [0] * (k + 1)
    for i in range(1, k):
        j = pi[i - 1]
        while j > 0 and bits[i] != bits[j]:
            j = pi[j - 1]
        if bits[i] == bits[j]:
            j += 1
        pi[i] = j

    def go(s, b):
        t = s
        while t > 0 and (t == k or bits[t] != b):
            t = pi[t - 1]
        if t < k and bits[t] == b:
            t += 1
        return t

    return k, go


def contains(n, word):
    k, go = _make_matcher(word)
    trans = {}
    for s in range(k + 1):
        for b in (0, 1):
            trans[(s, b)] = (k,) if s == k else (go(s, b),)
    return n, k + 1, 0, [k], trans


def ends_with(n, word):
    k, go = _make_matcher(word)
    trans = {}
    for s in range(k + 1):
        for b in (0, 1):
            trans[(s, b)] = (go(s, b),)
    return n, k + 1, 0, [k], trans


def random_dfa(rng, n, S, K):
    trans = {}
    for p in range(S):
        for b in (0, 1):
            trans[(p, b)] = (rng.randrange(S),)
    pool = [s for s in range(1, S)]
    accepts = sorted(rng.sample(pool, min(K, len(pool))))
    return n, S, 0, accepts, trans


def add_clones(n, S, q0, accepts, trans, clone_states):
    S2 = S
    trans2 = {k: set(v) for k, v in trans.items()}
    accepts2 = set(accepts)
    for p in clone_states:
        p2 = S2
        S2 += 1
        for k2 in list(trans2.keys()):
            if p in trans2[k2]:
                trans2[k2].add(p2)
        for b in (0, 1):
            trans2[(p2, b)] = set(trans2.get((p, b), set()))
        if p in accepts2:
            accepts2.add(p2)
    trans2 = {k: tuple(sorted(v)) for k, v in sorted(trans2.items())}
    return n, S2, q0, sorted(accepts2), trans2

# ---------------------------------------------------------------- measurement

def dfa_stats(n, S, q0, accepts, trans):
    """Exact (|L|, total 0-bits over L) for a DFA (paths == strings)."""
    cnt = [0] * S
    zer = [0] * S
    cnt[q0] = 1
    for _ in range(n):
        nc = [0] * S
        nz = [0] * S
        for p in range(S):
            c = cnt[p]
            if not c:
                continue
            z = zer[p]
            for b in (0, 1):
                for q in trans.get((p, b), ()):
                    nc[q] += c
                    nz[q] += z + (c if b == 0 else 0)
        cnt, zer = nc, nz
    A = sum(cnt[a] for a in accepts)
    sumz = sum(zer[a] for a in accepts)
    return A, sumz


def measure(inst):
    """Returns (A, B, F_s, r_s) for an instance (n,S,q0,accepts,trans)."""
    n, S, q0, accepts, trans = inst
    minterms = _greedy.accepted_strings(n, q0, accepts, trans)
    A = len(minterms)
    sumz = sum(n - bin(v).count("1") for v in minterms)
    B = sumz + A * (n - 1) + max(0, A - 1)
    gates_s, _ = _strong.build_circuit(n, S, q0, accepts, trans)
    F_s = max(1, len(gates_s))
    r_s = min(1.0, 0.1 * B / F_s)
    return A, B, F_s, r_s


def measure_greedy(inst, B):
    n, S, q0, accepts, trans = inst
    minterms = _greedy.accepted_strings(n, q0, accepts, trans)
    terms = _greedy.minimize_cover(n, minterms)
    gates_g, _ = _greedy.emit_circuit(n, terms)
    F_g = max(1, len(gates_g))
    return F_g, min(1.0, 0.1 * B / F_g)

# ---------------------------------------------------------------- test ladder
# band_s / band_g are acceptance windows for the PREDICTED ratios
# (0.1*B/F). Random tests rejection-sample until both predictions land.

TESTS = {
    1: dict(kind="popcount_exact", n=12, k=1, clones=1,
            band_s=(0.42, 0.90), band_g=(0.13, 0.45)),
    2: dict(kind="popcount_exact", n=8, k=2, clones=0,
            band_s=(0.42, 0.90), band_g=(0.13, 0.45)),
    3: dict(kind="popcount_mod", n=9, m=6, r=2, clones=1,
            band_s=(0.42, 0.90), band_g=(0.13, 0.45)),
    4: dict(kind="random", n=10, S=10, K=1, clones=1,
            band_s=(0.50, 0.62), band_g=(0.13, 0.45)),
    5: dict(kind="random", n=11, S=13, K=1, clones=0,
            band_s=(0.58, 0.70), band_g=(0.13, 0.45)),
    6: dict(kind="random", n=11, S=12, K=1, clones=2,
            band_s=(0.64, 0.76), band_g=(0.13, 0.45)),
    7: dict(kind="random", n=12, S=17, K=1, clones=0,
            band_s=(0.70, 0.82), band_g=(0.13, 0.45)),
    8: dict(kind="random", n=12, S=15, K=1, clones=3,
            band_s=(0.78, 0.90), band_g=(0.13, 0.45)),
    9: dict(kind="random", n=11, S=9, K=2, clones=0,
            band_s=(0.54, 0.68), band_g=(0.13, 0.45)),
    10: dict(kind="random", n=12, S=13, K=2, clones=2,
            band_s=(0.78, 0.90), band_g=(0.13, 0.45)),
}

A_WINDOW = (8, 2200)


def build_structured(cfg):
    kind = cfg["kind"]
    if kind == "popcount_exact":
        inst = popcount_exact(cfg["n"], cfg["k"])
    elif kind == "popcount_mod":
        inst = popcount_mod(cfg["n"], cfg["m"], cfg["r"])
    elif kind == "contains":
        inst = contains(cfg["n"], cfg["word"])
    elif kind == "ends_with":
        inst = ends_with(cfg["n"], cfg["word"])
    else:
        raise ValueError(kind)
    return inst


def emit(inst):
    n, S, q0, accepts, trans = inst
    trlines = [(p, b, q) for (p, b) in sorted(trans.keys())
               for q in trans[(p, b)]]
    lines = ["%d %d %d %d %d" % (n, S, q0, len(accepts), len(trlines))]
    lines.append(" ".join(str(a) for a in accepts))
    for (p, b, q) in trlines:
        lines.append("%d %d %d" % (p, b, q))
    return "\n".join(lines) + "\n"


def reparsed_measure(text):
    """Parse the EMITTED text back and measure it -- guards against any
    emit/parse mismatch (e.g. transition-line count drift)."""
    toks = text.split()
    pos = [0]

    def nxt():
        t = int(toks[pos[0]]); pos[0] += 1
        return t

    n = nxt(); S = nxt(); q0 = nxt(); K = nxt(); M = nxt()
    accepts = [nxt() for _ in range(K)]
    trans = {}
    for _ in range(M):
        p = nxt(); b = nxt(); q = nxt()
        trans.setdefault((p, b), set()).add(q)
    assert pos[0] == len(toks), "trailing tokens after reparse"
    trans = {k: tuple(sorted(v)) for k, v in trans.items()}
    return measure((n, S, q0, accepts, trans))


def main():
    tid = int(sys.argv[1])
    cfg = TESTS[tid]
    slo, shi = cfg["band_s"]
    glo, ghi = cfg["band_g"]
    rng = random.Random(910_000_013 + 7919 * tid)

    if cfg["kind"] != "random":
        inst = build_structured(cfg)
        base = inst
        if cfg.get("clones"):
            n, S, q0, accepts, trans = inst
            pool = [s for s in range(1, S)]
            clones = sorted(rng.sample(pool, min(cfg["clones"], len(pool))))
            inst = add_clones(n, S, q0, accepts, trans, clones)
        A, B, F_s, r_s = measure(inst)
        if not (A_WINDOW[0] <= A <= A_WINDOW[1] and slo <= r_s <= shi):
            # fall back to the un-cloned automaton, re-check
            inst = base
            A, B, F_s, r_s = measure(inst)
            assert A_WINDOW[0] <= A <= A_WINDOW[1], (tid, A)
            assert slo <= r_s <= shi, (tid, r_s)
        F_g, r_g = measure_greedy(inst, B)
        assert glo <= r_g <= ghi, (tid, r_g)
        sys.stdout.write(emit(inst))
        return

    # random: seeded rejection sampling against predicted ladder ratios
    best = None
    for attempt in range(6000):
        inst = random_dfa(rng, cfg["n"], cfg["S"], cfg["K"])
        A, sumz = dfa_stats(inst[0], inst[1], inst[2], inst[3], inst[4])
        if not (A_WINDOW[0] <= A <= A_WINDOW[1]):
            continue
        B = sumz + A * (inst[0] - 1) + max(0, A - 1)
        gates_s, _ = _strong.build_circuit(*inst)
        F_s = max(1, len(gates_s))
        r_s = min(1.0, 0.1 * B / F_s)
        if not (slo <= r_s <= shi):
            continue
        F_g, r_g = measure_greedy(inst, B)
        if not (glo <= r_g <= ghi):
            if best is None or abs(r_g - (glo + ghi) / 2) < best[0]:
                best = (abs(r_g - (glo + ghi) / 2), inst)
            continue
        break
    else:
        assert best is not None, tid
        inst = best[1]

    if cfg.get("clones"):
        n, S, q0, accepts, trans = inst
        A0, B0, F0, r0 = measure(inst)
        pool = [s for s in range(1, S)]
        clones = sorted(rng.sample(pool, min(cfg["clones"], len(pool))))
        cand = add_clones(n, S, q0, accepts, trans, clones)
        A1, B1, F1, r1 = measure(cand)
        # clones must not change the language or the strong gate count
        if A1 == A0 and F1 == F0:
            inst = cand
    sys.stdout.write(emit(inst))


if __name__ == "__main__":
    main()

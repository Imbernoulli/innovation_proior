#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  (ans ignored) -- deterministic checker for the
regex-engine-repair ("automaton-patching") problem.

Reads testId from <in>'s first line and REGENERATES (identically to gen.py,
same seeded construction) both the hidden CORRECT automaton and the damaged
one shown to the solver -- the correct automaton and the planted defect are
never read from <in>; they live only in code here (and in gen.py, duplicated).

Parses the participant's edit list from <out>, validates it strictly:
  - well-formed opcodes/args, all state ids in range, symbols in {0,1}
  - total edit cost (1 per edit line) <= the budget printed in <in>
Applies the edits to the DAMAGED automaton, then regenerates a large
HELD-OUT suite of (string, expected-label) pairs -- disjoint from the visible
examples, using the same generator family PLUS a systematic sweep over every
motif-boundary and every intra-star-loop truncation point (the sweep is what
makes a narrow planted defect reliably observable rather than a matter of
sampling luck) -- and scores classification accuracy F on that suite.

Baseline B is a fixed, calibrated constant (0.30): the checker's own trivial
"pessimistic reject-leaning" reference for this family. Score:
    Ratio = min(1.0, 0.1 * F / B)
A no-edit submission (F = raw damaged-automaton accuracy, typically well
under B) scores near/under the ~0.1 convention; exact root-cause repair
(F ~= 1.0) lands around 0.33, leaving headroom above it.
"""
import sys, random

MOTIF_POOL = ['LIT0', 'LIT1', 'STAR0', 'STAR1', 'OPT01']
STAR_MAX = 10
B_BASELINE = 0.30
N_HELD_TARGET = 200
SWEEP_INSTANTIATIONS = 35
SWEEP_CAP = 45
MAX_STATES = 500
MAX_EDITS = 2000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------------- hidden construction (identical to gen.py) ----------------

def _add_eps(eps_adj, a, b):
    eps_adj.setdefault(a, set()).add(b)


def _build_lit(entry, c, trans, eps_adj, counter):
    exitst = counter[0]; counter[0] += 1
    trans[(entry, c)] = exitst
    return exitst


def _build_star(entry, c, trans, eps_adj, counter, star_edges, motif_idx):
    loop = counter[0]; counter[0] += 1
    loop2 = counter[0]; counter[0] += 1
    exitst = counter[0]; counter[0] += 1
    _add_eps(eps_adj, entry, loop)
    _add_eps(eps_adj, entry, exitst)
    trans[(loop, c)] = loop2
    star_edges.append((loop, c, loop2, motif_idx))
    _add_eps(eps_adj, loop2, loop)
    _add_eps(eps_adj, loop2, exitst)
    return exitst


def _build_opt01(entry, trans, eps_adj, counter):
    mid1 = counter[0]; counter[0] += 1
    mid2 = counter[0]; counter[0] += 1
    exitst = counter[0]; counter[0] += 1
    _add_eps(eps_adj, entry, exitst)
    _add_eps(eps_adj, entry, mid1)
    trans[(mid1, '0')] = mid2
    trans[(mid2, '1')] = exitst
    return exitst


def _family(m):
    if m in ('LIT0', 'STAR0'):
        return '0'
    if m in ('LIT1', 'STAR1'):
        return '1'
    return 'OPT'


def build_correct(test_id):
    rng = random.Random("motifs-%d" % test_id)
    K = 4 + (test_id - 1) // 3
    for _try in range(30):
        motifs = []
        prev_fam = None
        for _ in range(K):
            cands = [m for m in MOTIF_POOL if _family(m) != prev_fam]
            m = rng.choice(cands)
            motifs.append(m)
            prev_fam = _family(m)
        has_star = any(mm.startswith('STAR') for mm in motifs)
        has_qual_boundary = any(
            any(mm.startswith('LIT') for mm in motifs[idx + 1:])
            for idx in range(K - 1))
        if has_star and has_qual_boundary:
            break

    trans = {}
    eps_adj = {}
    counter = [1]
    star_edges = []
    boundaries = []
    cur = 0
    final = 0
    for i, m in enumerate(motifs):
        entry = cur
        if m == 'LIT0':
            nxt = _build_lit(entry, '0', trans, eps_adj, counter)
        elif m == 'LIT1':
            nxt = _build_lit(entry, '1', trans, eps_adj, counter)
        elif m == 'STAR0':
            nxt = _build_star(entry, '0', trans, eps_adj, counter, star_edges, i)
        elif m == 'STAR1':
            nxt = _build_star(entry, '1', trans, eps_adj, counter, star_edges, i)
        else:
            nxt = _build_opt01(entry, trans, eps_adj, counter)
        if i < K - 1:
            j = counter[0]; counter[0] += 1
            _add_eps(eps_adj, nxt, j)
            boundaries.append((nxt, j))
            cur = j
        else:
            final = nxt
    n = counter[0]
    automaton = {'n': n, 'start': 0, 'accept': {final}, 'trans': trans, 'eps': eps_adj}
    meta = {'boundaries': boundaries, 'star_edges': star_edges, 'final': final, 'motifs': motifs}
    return automaton, meta


DEFECT_SCHEDULE = ['D1', 'D2', 'D3', 'D1', 'D3', 'D2', 'D3', 'D1', 'D3', 'D2']


def apply_defect(automaton, meta, test_id):
    dtype = DEFECT_SCHEDULE[(test_id - 1) % len(DEFECT_SCHEDULE)]
    rng = random.Random("defect-%d" % test_id)
    trans = dict(automaton['trans'])
    eps_adj = {k: set(v) for k, v in automaton['eps'].items()}
    accept = set(automaton['accept'])
    n = automaton['n']
    if dtype == 'D1':
        a, b = rng.choice(meta['boundaries'])
        eps_adj[a].discard(b)
        if not eps_adj[a]:
            del eps_adj[a]
        root_cause = ('ADDEPS', a, b)
    elif dtype == 'D2':
        loop, c, correct_target, _idx = rng.choice(meta['star_edges'])
        trap = n; n += 1
        trans[(loop, c)] = trap
        root_cause = ('SETTRANS', loop, c, correct_target)
    else:
        motifs = meta['motifs']
        star_qual = [(l2, idx) for (_l, c, l2, idx) in meta['star_edges']
                     if any(mm == ('LIT1' if c == '0' else 'LIT0')
                            for mm in motifs[idx + 1:])]
        if not star_qual:
            star_qual = [(l2, idx) for (_l, _c, l2, idx) in meta['star_edges']
                         if any(mm.startswith('LIT') for mm in motifs[idx + 1:])]
        if star_qual:
            a, _idx = rng.choice(star_qual)
        else:
            qualifying = [a for idx, (a, b) in enumerate(meta['boundaries'])
                          if any(mm.startswith('LIT') for mm in motifs[idx + 1:])]
            pool = qualifying if qualifying else [a for a, b in meta['boundaries']]
            a = rng.choice(pool)
        accept.add(a)
        root_cause = ('TOGGLE', a)
    broken = {'n': n, 'start': automaton['start'], 'accept': accept, 'trans': trans, 'eps': eps_adj}
    return broken, dtype, root_cause


def eps_closure(states, eps_adj):
    stack = list(states)
    seen = set(states)
    while stack:
        s = stack.pop()
        for t in eps_adj.get(s, ()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def run_nfa(automaton, string):
    cur = eps_closure({automaton['start']}, automaton['eps'])
    trans = automaton['trans']
    for ch in string:
        nxt = set()
        for s in cur:
            t = trans.get((s, ch))
            if t is not None:
                nxt.add(t)
        cur = eps_closure(nxt, automaton['eps'])
    return any(s in automaton['accept'] for s in cur)


def sample_positive_with_bounds(motifs, rng, star_max=STAR_MAX):
    parts = []
    bounds = []
    checkpoints = []
    total = 0
    for m in motifs:
        if m == 'LIT0':
            seg = '0'
        elif m == 'LIT1':
            seg = '1'
        elif m == 'STAR0':
            k = rng.randint(0, star_max)
            seg = '0' * k
            for it in range(1, k + 1):
                checkpoints.append(total + it)
        elif m == 'STAR1':
            k = rng.randint(0, star_max)
            seg = '1' * k
            for it in range(1, k + 1):
                checkpoints.append(total + it)
        else:
            seg = '01' if rng.random() < 0.5 else ''
        parts.append(seg)
        total += len(seg)
        bounds.append(total)
    return ''.join(parts), bounds, checkpoints


def sample_positive(motifs, rng):
    return sample_positive_with_bounds(motifs, rng)[0]


def mutate(s, rng):
    if not s:
        return '0' if rng.random() < 0.5 else '1'
    kind = rng.randrange(3)
    i = rng.randrange(len(s))
    if kind == 0:
        c = '1' if s[i] == '0' else '0'
        return s[:i] + c + s[i + 1:]
    elif kind == 1:
        return s[:i] + s[i + 1:]
    else:
        return s[:i] + s[i] + s[i:]


def random_string(rng, max_len=12):
    L = rng.randint(1, max_len)
    return ''.join(rng.choice('01') for _ in range(L))


def _one_negative_attempt(motifs, correct, rng):
    full, bounds, checkpoints = sample_positive_with_bounds(motifs, rng)
    trunc_pts = list(bounds[:-1]) + checkpoints
    r2 = rng.random()
    if r2 < 0.85 and trunc_pts:
        L = rng.choice(trunc_pts)
        s = full[:L]
    elif r2 < 0.95:
        s = mutate(full, rng)
    else:
        s = random_string(rng)
    return s, run_nfa(correct, s)


def sample_labeled(motifs, correct, rng, want_pos_frac=0.65):
    if rng.random() < want_pos_frac:
        s = sample_positive(motifs, rng)
        if s:
            return s, True
    for _ in range(8):
        s, lab = _one_negative_attempt(motifs, correct, rng)
        if not lab and s:
            return s, False
    s = sample_positive(motifs, rng)
    return (s, True) if s else ('1', run_nfa(correct, '1'))


def gen_examples(motifs, corr, broken, rng, k_want, max_tries=6000):
    out = []
    seenset = set()
    tries = 0
    while len(out) < k_want and tries < max_tries:
        tries += 1
        s, lab = sample_labeled(motifs, corr, rng)
        if s in seenset:
            continue
        if run_nfa(broken, s) != lab:
            out.append((s, lab))
            seenset.add(s)
    return out


def systematic_prefix_sweep(motifs, correct, rng, n_instantiations=SWEEP_INSTANTIATIONS):
    out = []
    for _ in range(n_instantiations):
        full, bounds, checkpoints = sample_positive_with_bounds(motifs, rng)
        pts = sorted(set(bounds[:-1]) | set(checkpoints))
        for L in pts:
            s = full[:L]
            if s and not run_nfa(correct, s):
                out.append((s, False))
    return out


def budget_for(test_id):
    return 6 + (test_id % 2)


# ---------------- read <in> ----------------

def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    tid = int(nxt())
    n = int(nxt())
    k_acc = int(nxt())
    accept = set(int(nxt()) for _ in range(k_acc))
    m_eps = int(nxt())
    eps = []
    for _ in range(m_eps):
        a = int(nxt()); b = int(nxt())
        eps.append((a, b))
    m_trans = int(nxt())
    trans = []
    for _ in range(m_trans):
        a = int(nxt()); c = nxt(); b = int(nxt())
        trans.append((a, c, b))
    budget = int(nxt())
    k_ex = int(nxt())
    examples = []
    for _ in range(k_ex):
        s = nxt(); lab = int(nxt())
        examples.append((s, lab))
    return tid, n, accept, eps, trans, budget, examples


# ---------------- read & validate <out> ----------------

def read_edits(path, n, budget):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")
    if not toks:
        fail("empty output")
    it = iter(toks)

    def nxt():
        return next(it)

    try:
        K = int(nxt())
    except (StopIteration, ValueError):
        fail("missing/invalid edit count")
    if K < 0 or K > MAX_EDITS:
        fail("edit count out of range")
    edits = []
    try:
        for _ in range(K):
            op = nxt().upper()
            if op == 'TOGGLE':
                s = int(nxt())
                if not (0 <= s < n):
                    fail("TOGGLE state out of range")
                edits.append(('TOGGLE', s))
            elif op == 'ADDEPS' or op == 'DELEPS':
                a = int(nxt()); b = int(nxt())
                if not (0 <= a < n and 0 <= b < n):
                    fail("%s state out of range" % op)
                edits.append((op, a, b))
            elif op == 'SETTRANS':
                a = int(nxt()); c = nxt(); b = int(nxt())
                if c not in ('0', '1'):
                    fail("SETTRANS symbol must be 0/1")
                if not (0 <= a < n and 0 <= b < n):
                    fail("SETTRANS state out of range")
                edits.append(('SETTRANS', a, c, b))
            else:
                fail("unknown opcode %r" % op)
    except (StopIteration, ValueError):
        fail("malformed edit line")
    # trailing garbage tokens are ignored (only the first K edits count),
    # matching the "print the score on the LAST Ratio line" style tolerance
    cost = len(edits)
    if cost > budget:
        fail("edit cost %d exceeds budget %d" % (cost, budget))
    return edits


def apply_edits(n, accept, eps_adj, trans, edits):
    acc = set(accept)
    eps2 = {a: set(v) for a, v in eps_adj.items()}
    tr2 = dict(trans)
    for e in edits:
        if e[0] == 'TOGGLE':
            s = e[1]
            if s in acc:
                acc.discard(s)
            else:
                acc.add(s)
        elif e[0] == 'ADDEPS':
            eps2.setdefault(e[1], set()).add(e[2])
        elif e[0] == 'DELEPS':
            if e[1] in eps2:
                eps2[e[1]].discard(e[2])
        elif e[0] == 'SETTRANS':
            tr2[(e[1], e[2])] = e[3]
    return acc, eps2, tr2


def accuracy(start, accept, eps_adj, trans, samples):
    if not samples:
        return 1.0
    aut = {'start': start, 'accept': accept, 'eps': eps_adj, 'trans': trans}
    c = 0
    for s, lab in samples:
        pred = run_nfa(aut, s)
        if pred == bool(lab):
            c += 1
    return c / len(samples)


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]
    tid, n, accept_in, eps_in, trans_in, budget, examples = read_instance(in_path)
    if n <= 0 or n > MAX_STATES:
        fail("bad instance (internal)")

    # rebuild eps/trans dicts as given in <in> (== the damaged automaton)
    eps_adj = {}
    for a, b in eps_in:
        eps_adj.setdefault(a, set()).add(b)
    trans = {}
    for a, c, b in trans_in:
        trans[(a, c)] = b

    edits = read_edits(out_path, n, budget)
    acc2, eps2, tr2 = apply_edits(n, accept_in, eps_adj, trans, edits)

    # regenerate the hidden correct automaton + held-out suite (never from <in>)
    corr, meta = build_correct(tid)
    motifs = meta['motifs']
    # reproduce the exact visible-example draw so held-out can exclude it
    broken0, _dtype, _rc = apply_defect(corr, meta, tid)
    vis_rng = random.Random("vis-%d" % tid)
    visible = gen_examples(motifs, corr, broken0, vis_rng, 5)
    vis_set = set(s for s, _ in visible)

    held_rng = random.Random("held-%d" % tid)
    held = []
    sweep = systematic_prefix_sweep(motifs, corr, held_rng)
    held_rng.shuffle(sweep)
    for s, lab in sweep:
        if s not in vis_set and len(held) < SWEEP_CAP:
            held.append((s, lab))
    tries = 0
    while len(held) < N_HELD_TARGET and tries < 8000:
        tries += 1
        s, lab = sample_labeled(motifs, corr, held_rng)
        if s in vis_set:
            continue
        held.append((s, lab))

    F = accuracy(0, acc2, eps2, tr2, held)
    if F != F or F in (float('inf'), float('-inf')):
        fail("non-finite score")
    sc = min(1000.0, 100.0 * F / max(1e-9, B_BASELINE))
    ratio = sc / 1000.0
    print("edits_used=%d budget=%d held_out_accuracy=%.6f  Ratio: %.6f"
          % (len(edits), budget, F, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()

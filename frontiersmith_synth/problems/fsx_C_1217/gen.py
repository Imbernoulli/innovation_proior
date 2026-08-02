#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE regex-engine-repair instance to stdout.

A small NFA (states 0..n-1, alphabet {0,1}, start=0) was built by concatenating
K "motifs" (Thompson-style: literal / Kleene-star / optional-literal fragments)
glued by epsilon edges. ONE structural defect was then planted -- exactly one
of: a missing epsilon glue edge, a star's loop transition rerouted into a dead
trap state, or one extra internal state wrongly marked accepting. That single
defect is never printed; only its damaged automaton + a handful of currently
MISCLASSIFIED (string, expected-label) examples are.

The hidden "correct" automaton, the defect type/location, and the held-out
grading suite live ONLY in verify.py (regenerated there, deterministically,
from the same testId) -- never printed here.

STDOUT:
  testId
  n_states
  k_accept
  <k_accept accepting state ids>
  m_eps
  <m_eps lines: a b>                  (epsilon edge a -> b)
  m_trans
  <m_trans lines: a c b>              (on symbol c in {0,1}, a -> b)
  budget
  k_examples
  <k_examples lines: string label>    (label 1 = should ACCEPT, 0 = should REJECT;
                                        both are currently MISCLASSIFIED by the
                                        automaton above)
"""
import sys, random

MOTIF_POOL = ['LIT0', 'LIT1', 'STAR0', 'STAR1', 'OPT01']
STAR_MAX = 10


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
    # Adjacent motifs must differ in symbol-family, else one silently absorbs
    # the other's contribution to the language and a planted defect becomes
    # unobservable (e.g. two consecutive STAR1's are redundant with each other).
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
        # Plant the false accept INSIDE a star's loop-body state (revisited at
        # every iteration count, so many string lengths land there -- broad
        # exposure, not one exact length). Require a MANDATORY literal of the
        # OPPOSITE symbol later on, so a truncated prefix stopping mid-loop
        # can't be confused with an unrelated minimal parse that legitimately
        # supplies that literal (same-symbol confusability hides the defect).
        star_qual = [(loop2, idx) for (_l, c, loop2, idx) in
                     [(l, c, l2, ix) for (l, c, l2, ix) in meta['star_edges']]
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


def budget_for(test_id):
    return 6 + (test_id % 2)


def main():
    tid = int(sys.argv[1])
    corr, meta = build_correct(tid)
    broken, dtype, root_cause = apply_defect(corr, meta, tid)
    vis_rng = random.Random("vis-%d" % tid)
    visible = gen_examples(meta['motifs'], corr, broken, vis_rng, 5)

    n = broken['n']
    accept = sorted(broken['accept'])
    eps_lines = []
    for a in sorted(broken['eps']):
        for b in sorted(broken['eps'][a]):
            eps_lines.append((a, b))
    trans_lines = []
    for (a, c) in sorted(broken['trans']):
        trans_lines.append((a, c, broken['trans'][(a, c)]))
    budget = budget_for(tid)

    out = []
    out.append(str(tid))
    out.append(str(n))
    out.append(str(len(accept)))
    out.append(' '.join(map(str, accept)) if accept else '')
    out.append(str(len(eps_lines)))
    for a, b in eps_lines:
        out.append("%d %d" % (a, b))
    out.append(str(len(trans_lines)))
    for a, c, b in trans_lines:
        out.append("%d %s %d" % (a, c, b))
    out.append(str(budget))
    out.append(str(len(visible)))
    for s, lab in visible:
        out.append("%s %d" % (s, 1 if lab else 0))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == "__main__":
    main()

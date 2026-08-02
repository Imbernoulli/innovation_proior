# TIER: greedy
"""The obvious per-example patch: for each visible failing example, run it
through the automaton and see exactly where matching goes wrong -- then mark
THAT point as the new boundary (toggle whether it counts as accepting).

This is a real, principled-looking fix (not a random guess) and it always
resolves the example it was aimed at. But it is symptom-patching: each edit
is calibrated to one specific example's run, not to the shared structural
cause, so it transfers only to the (typically few) OTHER strings that happen
to end their run at the exact same state -- and can just as easily introduce
new mistakes on strings that pass through that state differently.
"""
import sys


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


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    tid = int(nxt())
    n = int(nxt())
    k_acc = int(nxt())
    accept = set(int(nxt()) for _ in range(k_acc))
    m_eps = int(nxt())
    eps_adj = {}
    for _ in range(m_eps):
        a = int(nxt()); b = int(nxt())
        eps_adj.setdefault(a, set()).add(b)
    m_trans = int(nxt())
    trans = {}
    for _ in range(m_trans):
        a = int(nxt()); c = nxt(); b = int(nxt())
        trans[(a, c)] = b
    budget = int(nxt())
    k_ex = int(nxt())
    examples = []
    for _ in range(k_ex):
        s = nxt(); lab = int(nxt())
        examples.append((s, lab == 1))
    return n, accept, eps_adj, trans, budget, examples


def main():
    n, accept, eps_adj, trans, budget, examples = read_instance()
    acc = set(accept)
    edits = []
    for s, expected in examples:
        if len(edits) >= budget:
            break
        cur = eps_closure({0}, eps_adj)
        for ch in s:
            nxt_states = set()
            for st in cur:
                t = trans.get((st, ch))
                if t is not None:
                    nxt_states.add(t)
            cur = eps_closure(nxt_states, eps_adj)
        predicted = any(st in acc for st in cur)
        if predicted == expected:
            continue
        if expected and not predicted:
            if not cur:
                continue  # the run died with nowhere to land; nothing local to patch
            cand = max(cur)
            if cand not in acc:
                acc.add(cand)
                edits.append(('TOGGLE', cand))
        else:
            hit = [st for st in cur if st in acc]
            if hit:
                cand = max(hit)
                acc.discard(cand)
                edits.append(('TOGGLE', cand))

    out = [str(len(edits))]
    for e in edits:
        out.append("TOGGLE %d" % e[1])
    print('\n'.join(out))


if __name__ == "__main__":
    main()

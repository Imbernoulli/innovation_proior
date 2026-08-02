# TIER: strong
"""The insight: don't patch examples one at a time -- search for the ONE
structural edit that explains ALL of the visible failures simultaneously.
The automaton is small, so brute-forcing every single candidate edit (toggle
one state's accept bit / add one epsilon edge / retarget one transition) and
keeping only the edit that makes every given example correct at once is
cheap, and it is exactly the exchange argument that finds the shared root
cause instead of a pile of per-example special cases.
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


def run_nfa(accept, eps_adj, trans, s):
    cur = eps_closure({0}, eps_adj)
    for ch in s:
        nxt = set()
        for st in cur:
            t = trans.get((st, ch))
            if t is not None:
                nxt.add(t)
        cur = eps_closure(nxt, eps_adj)
    return any(st in accept for st in cur)


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


def consistent(accept, eps_adj, trans, examples):
    return all(run_nfa(accept, eps_adj, trans, s) == lab for s, lab in examples)


def try_toggle(accept, eps_adj, trans, examples, n):
    for s in range(n):
        a2 = set(accept)
        if s in a2:
            a2.discard(s)
        else:
            a2.add(s)
        if consistent(a2, eps_adj, trans, examples):
            return ('TOGGLE', s)
    return None


def try_addeps(accept, eps_adj, trans, examples, n):
    for a in range(n):
        for b in range(n):
            if a == b or b in eps_adj.get(a, ()):
                continue
            e2 = {k: set(v) for k, v in eps_adj.items()}
            e2.setdefault(a, set()).add(b)
            if consistent(accept, e2, trans, examples):
                return ('ADDEPS', a, b)
    return None


def try_settrans(accept, eps_adj, trans, examples, n):
    for a in range(n):
        for c in ('0', '1'):
            for b in range(n):
                if trans.get((a, c)) == b:
                    continue
                t2 = dict(trans)
                t2[(a, c)] = b
                if consistent(accept, eps_adj, t2, examples):
                    return ('SETTRANS', a, c, b)
    return None


def main():
    n, accept, eps_adj, trans, budget, examples = read_instance()

    if not examples or consistent(accept, eps_adj, trans, examples):
        print(0)
        return

    found = None
    for finder in (try_toggle, try_addeps, try_settrans):
        found = finder(accept, eps_adj, trans, examples, n)
        if found is not None:
            break

    if found is None:
        # defensive fallback (shouldn't trigger given the problem's guarantee
        # of a single-edit root cause): fall back to a per-example patch.
        acc = set(accept)
        edits = []
        for s, expected in examples:
            if len(edits) >= budget:
                break
            cur = eps_closure({0}, eps_adj)
            for ch in s:
                nxt = set()
                for st in cur:
                    t = trans.get((st, ch))
                    if t is not None:
                        nxt.add(t)
                cur = eps_closure(nxt, eps_adj)
            predicted = any(st in acc for st in cur)
            if predicted == expected:
                continue
            if expected and not predicted and cur:
                cand = max(cur)
                acc.add(cand)
                edits.append(('TOGGLE', cand))
            elif not expected and predicted:
                hit = [st for st in cur if st in acc]
                if hit:
                    cand = max(hit)
                    acc.discard(cand)
                    edits.append(('TOGGLE', cand))
        print(len(edits))
        for e in edits:
            print("TOGGLE %d" % e[1])
        return

    print(1)
    if found[0] == 'TOGGLE':
        print("TOGGLE %d" % found[1])
    elif found[0] == 'ADDEPS':
        print("ADDEPS %d %d" % (found[1], found[2]))
    else:
        print("SETTRANS %d %s %d" % (found[1], found[2], found[3]))


if __name__ == "__main__":
    main()

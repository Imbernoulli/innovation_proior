# TIER: greedy
"""Textbook move: build the trie over the legal traces, then bottom-up merge any two
trie nodes that are indistinguishable USING ONLY the legal-continuation edges each of
them happens to have on record (classic minimal-acyclic-DFA / dictionary-automaton
construction -- a legitimate, well-known compaction). To turn the result into a total
transition table (every state needs a target for every symbol), any symbol a merged
state never saw in the training data is routed back to the START state -- "if it isn't
one of the moves I was shown, treat it as a fresh handshake attempt" -- instead of a
dedicated reject state.

This never mis-handles a legal trace (their own symbols are always on record) and it
performs the SAME legitimate suffix-sharing compaction as the strong solution. But an
attacker who splices one never-seen symbol into an otherwise legal-looking prefix walks
straight back to the start state and can then simply replay any legal trace to
completion -- accepted. Merging purely on acceptance behaviour is not enough; the
automaton must also be built to consistently REJECT after a downgrade.
"""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = int(data[pos])
        pos += 1
        return v

    m = nxt()
    L = []
    n_l = nxt()
    for _ in range(n_l):
        ln = nxt()
        L.append(tuple(nxt() for _ in range(ln)))
    n_f = nxt()
    for _ in range(n_f):
        ln = nxt()
        for _ in range(ln):
            nxt()

    children = [dict()]
    accept = [False]
    for tr in L:
        cur = 0
        for c in tr:
            if c in children[cur]:
                cur = children[cur][c]
            else:
                children.append(dict())
                accept.append(False)
                nid = len(children) - 1
                children[cur][c] = nid
                cur = nid
        accept[cur] = True

    n = len(children)
    order = []
    sys.setrecursionlimit(10000)

    def dfs(u):
        for c, v in children[u].items():
            dfs(v)
        order.append(u)

    dfs(0)

    class_of = [None] * n
    sig_to_class = {}
    rep_edges = {}
    rep_accept = {}
    next_class = 0
    for u in order:
        sig = (accept[u], tuple(sorted((c, class_of[v]) for c, v in children[u].items())))
        if sig not in sig_to_class:
            sig_to_class[sig] = next_class
            rep_edges[next_class] = {c: class_of[v] for c, v in children[u].items()}
            rep_accept[next_class] = accept[u]
            next_class += 1
        class_of[u] = sig_to_class[sig]

    N = next_class
    root_class = class_of[0]

    # relabel so state 0 in the output is the trie root's class
    remap = list(range(N))
    if root_class != 0:
        remap[0], remap[root_class] = remap[root_class], remap[0]
    inv = [0] * N
    for new_id, old_id in enumerate(remap):
        inv[old_id] = new_id

    acc_out = [False] * N
    delta_out = [[0] * m for _ in range(N)]
    for old_id in range(N):
        nid = inv[old_id]
        acc_out[nid] = rep_accept[old_id]
        for c in range(m):
            if c in rep_edges[old_id]:
                delta_out[nid][c] = inv[rep_edges[old_id][c]]
            else:
                delta_out[nid][c] = 0  # BUG: reset-to-start on an unseen symbol

    out = [str(N), " ".join("1" if a else "0" for a in acc_out)]
    for row in delta_out:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

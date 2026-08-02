# TIER: strong
"""The insight: minimize by merging on FULL transition behaviour -- what a state must
do on EVERY symbol, including the ones no legal trace ever exercises from there -- not
just on the legal continuations it happens to have seen.

Concretely: build the trie over the legal traces plus ONE extra, explicit, self-looping,
non-accepting sink state, and route every not-on-a-legal-prefix symbol there (this is
exactly solutions/trivial.py's construction -- always safe by itself, since it is a
correct, total DFA for exactly the language L, and every forbidden trace is by
construction not a member of L). THEN run textbook Moore/Hopcroft partition refinement
on this total automaton: two states merge only if they agree on their accept bit AND on
the class of every one of their m outgoing edges -- including the forced "off-script"
edges into the sink partition. That is what makes the merge provably safe: a state that
is reachable only mid-legal-trace and a state that also has to swallow an out-of-band
attack symbol can never collapse together unless their FULL future behaviour agrees, so
the resulting automaton keeps rejecting after any downgrade/replay/reorder/rollback
while still sharing every genuinely-identical continuation (e.g. two different
handshake heads that allow exactly the same set of completions collapse into one).
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

    # 1) trie + explicit sink (total, always-safe DFA for language L)
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

    n_trie = len(children)
    sink = n_trie
    N0 = n_trie + 1
    delta = [[0] * m for _ in range(N0)]
    for u in range(n_trie):
        for c in range(m):
            delta[u][c] = children[u].get(c, sink)
    for c in range(m):
        delta[sink][c] = sink
    acc0 = accept + [False]

    # 2) Moore partition refinement over the TOTAL automaton (all m symbols)
    classes = [1 if a else 0 for a in acc0]
    while True:
        sig_map = {}
        new_classes = [0] * N0
        for i in range(N0):
            sig = (classes[i],) + tuple(classes[delta[i][c]] for c in range(m))
            cid = sig_map.get(sig)
            if cid is None:
                cid = len(sig_map)
                sig_map[sig] = cid
            new_classes[i] = cid
        if len(sig_map) == max(classes) + 1 and new_classes == classes:
            break
        classes = new_classes

    N = max(classes) + 1
    root_class = classes[0]
    remap = list(range(N))
    if root_class != 0:
        remap[0], remap[root_class] = remap[root_class], remap[0]
    inv = [0] * N
    for new_id, old_id in enumerate(remap):
        inv[old_id] = new_id

    acc_out = [False] * N
    delta_out = [[0] * m for _ in range(N)]
    seen = [False] * N
    for i in range(N0):
        cls = inv[classes[i]]
        if not seen[cls]:
            seen[cls] = True
            acc_out[cls] = acc0[i]
            for c in range(m):
                delta_out[cls][c] = inv[classes[delta[i][c]]]

    out = [str(N), " ".join("1" if a else "0" for a in acc_out)]
    for row in delta_out:
        out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

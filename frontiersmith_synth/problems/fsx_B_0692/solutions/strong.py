# TIER: strong
"""Residency planning, not tree traversal.

The insight: a shared value is only worth a STORE/LOAD (cost 8+8) if some of
its reuses cross into a *different output's* evaluation -- those are
genuinely far apart in the instruction stream and nothing in the cheap
DUP/SWAP/OVER/ROT permutation group (which only reaches the top few stack
slots) can retrieve it.  If every reuse of a shared value happens while
still inside the SAME output's evaluation, it is usually still near the top
of the stack by the time it's needed again -- so instead of spilling it, we
leave a spare copy sitting on the stack (via DUP/OVER) and let a
1-instruction shuffle fetch it back to the top when needed, at a fraction
of memory's cost.

Classification (static, from DAG structure alone):
  - a value used >=2 times whose live uses are reachable from >=2 different
    outputs -> CROSS: store once, load on every later use (memory really is
    required here -- same as the greedy tier).
  - a value used >=2 times, all live uses inside ONE output's closure ->
    LOCAL: never touches memory; a lightweight simulation of the exact
    DUP/SWAP/OVER/ROT transition rules the checker uses tells us whenever
    a live copy is still within reach, and a single shuffle grabs it.

SAFETY: naively leaving a "spare" copy right after computing a value can
silently corrupt a still-in-flight sibling operand of the OP that is about
to consume it -- concretely, doing this while computing something in the
*second* (immediately-pre-OP) operand slot bumps the wrong element into
that OP's second argument. So we only ever leave a spare when producing a
value into a "safe" slot (the first operand of a pending OP, or a
top-level output) -- never into the slot immediately consumed next. For
ADD/MUL (commutative) we're additionally free to choose *which* operand
goes first, so we always give the safe slot to whichever child is the
local-shared one when possible; only SUB is forced to keep its given
order.  Fetching an already-resident spare (DUP/OVER/ROT, reaching up to
the 3rd-from-top) is always safe regardless of slot: each of those three
instructions individually preserves the one property every caller relies
on -- "whatever sat on top right before this fetch ends up at exactly
depth 1 right after it" -- so chains of fetches (and the OPs consuming
their results) compose correctly no matter how deeply nested.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    for _ in range(N):
        next(it)
    nodes = {}
    for i in range(1, M + 1):
        op = next(it); cl = next(it); cr = next(it)
        nodes[i] = (op, cl, cr)
    outs = [next(it) for _ in range(K)]

    sys.setrecursionlimit(1000000)

    # ---- live-node reachability ----
    live = set()

    def collect(ref):
        if ref[0] != "N":
            return
        idx = int(ref[1:])
        if idx in live:
            return
        live.add(idx)
        _op, cl2, cr2 = nodes[idx]
        collect(cl2); collect(cr2)

    for o in outs:
        collect(o)

    # ---- usage counts, restricted to live nodes ----
    usage = {}

    def bump(ref):
        usage[ref] = usage.get(ref, 0) + 1

    for idx in live:
        _op, cl2, cr2 = nodes[idx]
        bump(cl2); bump(cr2)
    for o in outs:
        bump(o)

    shared_nodes = set(idx for idx in live if usage.get("N%d" % idx, 0) >= 2)

    # ---- reachable-output sets (per shared/live node) ----
    reach = {}

    def mark(ref, j, seen):
        if ref in seen:
            return
        seen.add(ref)
        if ref[0] == "N":
            idx = int(ref[1:])
            reach.setdefault(idx, set()).add(j)
            _op, cl2, cr2 = nodes[idx]
            mark(cl2, j, seen)
            mark(cr2, j, seen)

    for j, o in enumerate(outs, 1):
        mark(o, j, set())

    cross_nodes_all = set(idx for idx in shared_nodes if len(reach.get(idx, ())) >= 2)
    local_nodes = shared_nodes - cross_nodes_all

    # A CROSS value still only deserves STORE/LOAD if that actually beats
    # paying its raw recompute cost on every reuse -- for a cheap/shallow
    # cross value (e.g. one combined leaf pair), STORE(8)+LOAD(8) can easily
    # exceed just rebuilding it from scratch each time. This mirrors the
    # same cost comparison the greedy tier makes, so strong's edge over
    # greedy on cross values comes purely from not being *forced* into a
    # bad spill the way greedy's blanket rule can be -- both make the smart
    # call there; the real edge is entirely in the local/shuffle case.
    size = {}
    for i in range(1, N + 1):
        size[("L", i)] = 1
    for i in range(1, M + 1):
        _op, cl2, cr2 = nodes[i]
        size[("N", i)] = size[(cl2[0], int(cl2[1:]))] + size[(cr2[0], int(cr2[1:]))] + 1

    cross_nodes = set()
    for idx in cross_nodes_all:
        u = usage.get("N%d" % idx, 0)
        s = size[("N", idx)]
        if s + 9 + 8 * (u - 1) < s * u:
            cross_nodes.add(idx)

    def is_local(ref):
        return ref[0] == "N" and int(ref[1:]) in local_nodes

    # ---- emission with an exact symbolic stack simulation ----
    lines = []
    # Each stack slot is (tag, uid): `tag` identifies which node's value it
    # holds, `uid` is a unique id for *this specific copy*. Reservations are
    # tracked by uid, not by position, so they stay correct even when ROT
    # relocates a slot (uids travel with their value; a fresh DUP/OVER copy
    # always gets a brand-new uid and is therefore never accidentally
    # protected/blocked by someone else's reservation).
    stack_model = []
    stored_slots = set()
    reserved = set()   # uids of operands some still-open OP is waiting on
    next_uid = [0]

    def new_uid():
        next_uid[0] += 1
        return next_uid[0]

    def depth_in_top(tag, maxd):
        n = len(stack_model)
        for d in range(min(maxd + 1, n)):
            t, u = stack_model[-(d + 1)]
            if t == tag and u not in reserved:
                return d
        return None

    def fetch(d, tag):
        if d == 0:
            lines.append("DUP")
            stack_model.append((tag, new_uid()))
        elif d == 1:
            lines.append("OVER")
            stack_model.append((tag, new_uid()))
        else:  # d == 2: ROT brings the 3rd-from-top to the top. This is a pure
            # reposition (no growth, no new uid -- the same copy just moves),
            # but it still preserves the key property that whatever was on
            # top *before* this call ends up at exactly depth 1 afterward
            # (same as DUP/OVER), so it composes safely with the surrounding
            # operand bookkeeping -- see module docstring. Any reserved uid
            # caught in the window simply moves with its slot; reservations
            # are checked by uid, not position, so this is always safe.
            lines.append("ROT")
            v0, v1, v2 = stack_model[-1], stack_model[-2], stack_model[-3]
            stack_model[-3], stack_model[-2], stack_model[-1] = v1, v0, v2
        assert stack_model[-1][0] == tag

    def visit(ref, leave_spare_ok):
        if ref[0] == "L":
            i = int(ref[1:])
            lines.append("PUSH %d" % i)
            stack_model.append((("L", i), new_uid()))
            return
        idx = int(ref[1:])
        tag = ("N", idx)
        if idx in stored_slots:
            lines.append("LOAD %d" % idx)
            stack_model.append((tag, new_uid()))
            return
        d = depth_in_top(tag, 2)   # DUP(0)/OVER(1)/ROT(2) -- the full cheap shuffle reach
        if d is not None:
            fetch(d, tag)
            return
        op, cl, cr = nodes[idx]
        first, second = cl, cr
        # Only hoist a LOCAL child into the safe "first" slot if it is about
        # to be freshly computed (i.e. not already resident): fetching an
        # already-resident copy is safe in *either* slot (see fetch's
        # docstring note), so hoisting it would gain nothing -- while
        # actively being harmful when it *is* about to be recomputed fresh
        # and its own descendants (reached while visiting the sibling,
        # itself possibly a deep spine recursion) independently want the
        # very same shared value: an outer reservation would then block
        # every nested occurrence from reusing it, forcing them all the way
        # back down to a fresh recompute. Only ever reserve a value in the
        # single call frame that is truly minting a brand-new copy.
        if op in ("ADD", "MUL") and is_local(cr) and not is_local(cl):
            cr_idx = int(cr[1:])
            if cr_idx not in stored_slots and depth_in_top(("N", cr_idx), 2) is None:
                first, second = cr, cl
        # "first" is safe *only* if the whole chain of enclosing "first" links
        # back up to the nearest safe boundary was itself safe -- propagate the
        # incoming flag rather than resetting it to True, otherwise a spare left
        # deep inside an unsafe ("second") subtree still bloats that subtree's
        # net stack effect beyond the +1 its own parent OP assumes.
        visit(first, leave_spare_ok)
        # first's freshly produced value is now a reserved operand for the OP
        # below -- reserve its uid so second's entire subtree (however deeply
        # nested) can never fetch-and-consume this specific copy out from
        # under us (a different, non-reserved copy of the same tag is still
        # fair game).
        my_uid = stack_model[-1][1]
        reserved.add(my_uid)
        visit(second, False)
        reserved.discard(my_uid)
        lines.append("OP %s" % op)
        stack_model.pop(); stack_model.pop()
        stack_model.append((tag, new_uid()))
        if idx in cross_nodes:
            lines.append("DUP")
            stack_model.append((tag, new_uid()))
            lines.append("STORE %d" % idx)
            stack_model.pop()
            stored_slots.add(idx)
        elif idx in local_nodes and leave_spare_ok:
            lines.append("DUP")
            stack_model.append((tag, new_uid()))

    for o in outs:
        visit(o, True)
        lines.append("OUTPUT")
        stack_model.pop()

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

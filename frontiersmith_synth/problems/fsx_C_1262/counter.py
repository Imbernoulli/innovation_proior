import sys
from collections import defaultdict

# Format D checker -- Spiking-Neuron Core Map verifier + op-count (spike-traversal
# + core-overhead) scorer.
#
# Input <in>:
#   N C_max T slot_cap fanout_budget slot_rate_budget INTER LOCAL OVER
#   M
#   M lines: u v rate      (directed synapse u->v firing at integer rate>=1)
#
# Output <out>:
#   N                       (echo)
#   N lines: core slot      (line i, 0-indexed, is neuron i's assignment)
#
# Feasibility (checked against the instance's own limits):
#   - every core in [0, C_max), every slot in [0, T)
#   - per-core neuron count <= T*slot_cap                  (time-multiplex capacity)
#   - per-core sum of out-degree <= fanout_budget           (fanout-constraint)
#   - per-(core,slot) neuron count <= slot_cap
#   - per-(core,slot) sum of IN-rate <= slot_rate_budget    (time-multiplex-slots
#     x spike-traffic-routing: a slot's physical delivery throughput is capped
#     regardless of where the spikes originate)
#
# Objective (minimize): F = OVER * (#active cores)
#                          + sum over edges (u,v,rate) with core(u)!=core(v): rate*INTER
#                          + sum over edges (u,v,rate) with core(u)==core(v) and
#                            slot(u)!=slot(v): rate*LOCAL
#                          (same core AND same slot: free -- true same-cycle delivery)
#
# Baseline B: the checker's own round-robin-across-all-cores construction (spread
# neurons core (i % C_max) with an LPT-by-in-rate slot fill) -- identical to
# solutions/trivial.py. Ratio = min(1, 0.1*B/F).

MAXTOK = 4_000_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def lpt_slots(core_neurons, inrate, slot_cap, T):
    order = sorted(core_neurons, key=lambda x: (-inrate[x], x))
    slot_load = [0] * T
    slot_count = [0] * T
    assign = {}
    for nid in order:
        best_s = None
        for s in range(T):
            if slot_count[s] < slot_cap:
                if best_s is None or slot_load[s] < slot_load[best_s]:
                    best_s = s
        if best_s is None:
            return None
        assign[nid] = best_s
        slot_load[best_s] += inrate[nid]
        slot_count[best_s] += 1
    return assign


def roundrobin_baseline(N, C_max, T, slot_cap, outdeg, inrate, fanout_budget):
    core_list = [[] for _ in range(C_max)]
    fanout_sum = [0] * C_max
    core_of = [-1] * N
    for i in range(N):
        c = i % C_max
        tries = 0
        while tries < C_max and not (len(core_list[c]) < T * slot_cap and fanout_sum[c] + outdeg[i] <= fanout_budget):
            c = (c + 1) % C_max
            tries += 1
        if tries >= C_max:
            return None
        core_list[c].append(i)
        fanout_sum[c] += outdeg[i]
        core_of[i] = c
    slot_of = [None] * N
    for members in core_list:
        if not members:
            continue
        a = lpt_slots(members, inrate, slot_cap, T)
        if a is None:
            return None
        for i, s in a.items():
            slot_of[i] = s
    return core_of, slot_of


def compute_cost(N, edges, core_of, slot_of, INTER, LOCAL, OVER):
    used = len(set(core_of))
    total = used * OVER
    for u, v, r in edges:
        if core_of[u] != core_of[v]:
            total += r * INTER
        elif slot_of[u] != slot_of[v]:
            total += r * LOCAL
    return total


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    it = iter(inp)

    def nxt_int():
        return int(next(it))

    try:
        N = nxt_int(); C_max = nxt_int(); T = nxt_int(); slot_cap = nxt_int()
        fanout_budget = nxt_int(); slot_rate_budget = nxt_int()
        INTER = nxt_int(); LOCAL = nxt_int(); OVER = nxt_int()
    except Exception:
        fail("bad header")
    if not (1 <= N <= 2000 and 1 <= C_max <= 6000 and 1 <= T <= 32 and 1 <= slot_cap <= 4000
            and 0 <= fanout_budget <= 10**9 and 0 <= slot_rate_budget <= 10**9
            and 1 <= INTER <= 10**6 and 1 <= LOCAL <= 10**6 and 1 <= OVER <= 10**7):
        fail("bad dims")

    try:
        M = nxt_int()
    except Exception:
        fail("bad edge count")
    if not (0 <= M <= 400_000):
        fail("bad M")

    edges = []
    outdeg = [0] * N
    inrate = [0] * N
    try:
        for _ in range(M):
            u = nxt_int(); v = nxt_int(); r = nxt_int()
            if not (0 <= u < N and 0 <= v < N) or u == v:
                fail("bad edge endpoints")
            if not (1 <= r <= 10**6):
                fail("bad edge rate")
            edges.append((u, v, r))
            outdeg[u] += 1
            inrate[v] += r
    except Exception:
        fail("bad instance body")

    try:
        next(it)
        fail("trailing input tokens")
    except StopIteration:
        pass

    # ---- own baseline construction (guaranteed feasible by generator's margin) ----
    base = roundrobin_baseline(N, C_max, T, slot_cap, outdeg, inrate, fanout_budget)
    if base is None:
        fail("instance internal error: no feasible baseline (bad generator)")
    base_core_of, base_slot_of = base
    B = compute_cost(N, edges, base_core_of, base_slot_of, INTER, LOCAL, OVER)

    # ---- parse participant artifact ----
    if not out:
        fail("empty output")
    pos_tok = 0

    def take():
        nonlocal pos_tok
        if pos_tok >= len(out):
            raise IndexError
        v = out[pos_tok]
        pos_tok += 1
        return v

    try:
        n_out = int(take())
    except Exception:
        fail("bad n echo")
    if n_out != N:
        fail("n mismatch")

    core_of = [0] * N
    slot_of = [0] * N
    try:
        for i in range(N):
            c_tok = take(); s_tok = take()
            c = int(c_tok); s = int(s_tok)
            if not (0 <= c < C_max):
                fail("neuron %d: core id out of range" % i)
            if not (0 <= s < T):
                fail("neuron %d: slot id out of range" % i)
            core_of[i] = c
            slot_of[i] = s
    except IndexError:
        fail("truncated / wrong token count")
    except (ValueError, TypeError):
        fail("non-integer / non-finite token")

    if pos_tok != len(out):
        fail("trailing artifact tokens")

    # ---- feasibility: capacity, fanout-constraint, time-multiplex-slot rate budget ----
    core_count = defaultdict(int)
    core_fanout = defaultdict(int)
    slot_count = defaultdict(int)
    slot_rate = defaultdict(int)
    for i in range(N):
        c, s = core_of[i], slot_of[i]
        core_count[c] += 1
        core_fanout[c] += outdeg[i]
        slot_rate[(c, s)] += inrate[i]
        slot_count[(c, s)] += 1

    for c, cnt in core_count.items():
        if cnt > T * slot_cap:
            fail("core %d: neuron count %d exceeds capacity %d" % (c, cnt, T * slot_cap))
        if core_fanout[c] > fanout_budget:
            fail("core %d: fanout sum %d exceeds fanout_budget %d" % (c, core_fanout[c], fanout_budget))
    for (c, s), cnt in slot_count.items():
        if cnt > slot_cap:
            fail("core %d slot %d: neuron count %d exceeds slot_cap %d" % (c, s, cnt, slot_cap))
    for (c, s), rr in slot_rate.items():
        if rr > slot_rate_budget:
            fail("core %d slot %d: in-rate %d exceeds slot_rate_budget %d" % (c, s, rr, slot_rate_budget))

    F = compute_cost(N, edges, core_of, slot_of, INTER, LOCAL, OVER)
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("F=%d B=%d cores_used=%d Ratio: %.6f" % (F, B, len(core_count), ratio))


if __name__ == "__main__":
    main()

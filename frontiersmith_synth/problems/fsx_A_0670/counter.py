import sys

MAX_TOKEN_LEN = 15
MAX_ACTIONS = 2_000_000


def fail(reason):
    print("Ratio: 0.0  # %s" % reason)
    sys.exit(0)


def parse_int_token(tok):
    if len(tok) == 0 or len(tok) > MAX_TOKEN_LEN:
        return None
    t = tok
    if t[0] in "+-":
        t2 = t[1:]
    else:
        t2 = t
    if len(t2) == 0 or not t2.isdigit():
        return None
    try:
        return int(t)
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        fail("bad_args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_lines = f.read().split("\n")
    try:
        N, M, K = (int(x) for x in in_lines[0].split())
        costs_raw = in_lines[1].split()
        if len(costs_raw) != N:
            fail("bad_input_costs")
        costs = [0] * (N + 1)
        for i in range(1, N + 1):
            costs[i] = int(costs_raw[i - 1])
        if K > 0:
            reqs = [int(x) for x in in_lines[2].split()]
        else:
            reqs = []
        if len(reqs) != K:
            fail("bad_input_requests")
    except Exception:
        fail("unparseable_input")

    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + costs[i]

    try:
        with open(out_path, "r") as f:
            out_text = f.read()
    except Exception:
        fail("no_output")

    lines = [ln for ln in out_text.split("\n") if ln.strip() != ""]
    if len(lines) > MAX_ACTIONS:
        fail("too_many_actions")

    memory = set()
    served = []
    total_cost = 0

    for ln in lines:
        toks = ln.split()
        if len(toks) != 2:
            fail("malformed_line")
        op, itok = toks[0], toks[1]
        if op not in ("C", "E", "U"):
            fail("unknown_op")
        i = parse_int_token(itok)
        if i is None:
            fail("unparseable_index")
        if i < 1 or i > N:
            fail("index_out_of_range")

        if op == "C":
            if i > 1 and (i - 1) not in memory:
                fail("missing_dependency")
            total_cost += costs[i]
            memory.add(i)
            if len(memory) > M:
                fail("capacity_exceeded")
        elif op == "E":
            if i not in memory:
                fail("evict_non_resident")
            memory.discard(i)
        else:  # "U"
            if i not in memory:
                fail("serve_non_resident")
            served.append(i)

    if served != reqs:
        fail("serve_order_mismatch")

    # Internal baseline B: rebuild from node 1 for every request, using only
    # 2 live slots at a time (always feasible since M >= 2 by construction).
    B = sum(prefix[r] for r in reqs)

    F = total_cost
    if F <= 0:
        fail("nonpositive_cost")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("total_cost=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()

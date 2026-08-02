import sys

QTY_CAP = 5_000_000
PERIOD_CAP = 2_000_000


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_ints(tokens, k):
    out = []
    for _ in range(k):
        out.append(int(next(tokens)))
    return out


def parse_input(path):
    toks = iter(open(path).read().split())
    T, N, G = read_ints(toks, 3)
    V, P = read_ints(toks, 2)
    D = read_ints(toks, T)
    suppliers = []
    for _ in range(N):
        group, qualified, lead, qualcost, ntiers = read_ints(toks, 5)
        tiers = []
        for _ in range(ntiers):
            th, pr = read_ints(toks, 2)
            tiers.append((th, pr))
        tiers.sort()
        suppliers.append(dict(group=group, qualified=qualified, lead=lead,
                               qualcost=qualcost, tiers=tiers))
    E = int(next(toks))
    disruptions = set()
    for _ in range(E):
        per, grp = read_ints(toks, 2)
        disruptions.add((per, grp))
    return dict(T=T, N=N, G=G, V=V, P=P, D=D, suppliers=suppliers, disruptions=disruptions)


def tier_cost(supplier, q):
    if q <= 0:
        return 0
    price = supplier["tiers"][0][1]
    for th, pr in supplier["tiers"]:
        if q >= th:
            price = pr
        else:
            break
    return price * q


def evaluate_plan(inst, qual_actions, order_lines):
    """qual_actions: dict supplier_idx -> start_period (only for suppliers that
    start out unqualified). order_lines: list of (period, supplier_idx, qty).

    Each LINE is priced on its own quantity (no cross-line consolidation) --
    this is what makes concentrating an order into a single line the thing that
    actually claims a volume-discount tier; splitting one supplier's demand
    across several smaller lines in the same period only ever costs more,
    never less, since tier prices are non-increasing in quantity."""
    T, V, P, D = inst["T"], inst["V"], inst["P"], inst["D"]
    suppliers = inst["suppliers"]
    disruptions = inst["disruptions"]

    def available_from(i):
        s = suppliers[i]
        if s["qualified"]:
            return 1
        if i in qual_actions:
            return qual_actions[i] + s["lead"]
        return None  # never qualified

    avail = [available_from(i) for i in range(inst["N"])]

    delivered = [0.0] * (T + 1)
    cost = [0.0] * (T + 1)
    for (t, i, q) in order_lines:
        if q <= 0:
            continue
        s = suppliers[i]
        af = avail[i]
        if af is None or t < af:
            continue  # not yet qualified -- wasted order, no charge, no delivery
        if (t, s["group"]) in disruptions:
            continue  # disrupted -- wasted order, no charge, no delivery
        delivered[t] += q
        cost[t] += tier_cost(s, q)

    total = 0.0
    for t in range(1, T + 1):
        Dt = D[t - 1]
        deliv = delivered[t]
        total += min(Dt, deliv) * V - max(0.0, Dt - deliv) * P
        total -= cost[t]

    qual_cost_total = sum(suppliers[i]["qualcost"] for i in qual_actions)
    total -= qual_cost_total
    return total


def internal_baseline(inst):
    """Checker's own reference: single-source the primary supplier for the
    full demand every period, exactly like a diligent solver would -- except it
    places every order as a string of small unconsolidated line items (never
    noticing that batching one big order per period is what claims the volume
    discount) and never qualifies a second supplier. Fully meets demand
    whenever the primary is not disrupted, but pays close to the undiscounted
    price throughout."""
    T, D = inst["T"], inst["D"]
    chunk = 2
    lines = []
    for t in range(1, T + 1):
        Dt = D[t - 1]
        remaining = Dt
        while remaining > 0:
            q = min(chunk, remaining)
            lines.append((t, 0, q))
            remaining -= q
    return evaluate_plan(inst, {}, lines)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inst = parse_input(sys.argv[1])
    T, N = inst["T"], inst["N"]

    try:
        toks = iter(open(sys.argv[2]).read().split())
    except Exception:
        fail("cannot read output")

    try:
        Q = int(next(toks))
    except Exception:
        fail("missing Q")
    if not (0 <= Q <= N):
        fail("Q out of range")

    qual_actions = {}
    used_supplier = set()
    try:
        for _ in range(Q):
            i = int(next(toks))
            start = int(next(toks))
            if not (0 <= i < N):
                fail("qualification supplier index out of range")
            if inst["suppliers"][i]["qualified"]:
                fail("cannot 're-qualify' an already-qualified supplier")
            if i in used_supplier:
                fail("duplicate qualification action for a supplier")
            if not (1 <= start <= T):
                fail("qualification start period out of range")
            used_supplier.add(i)
            qual_actions[i] = start
    except SystemExit:
        raise
    except Exception:
        fail("bad qualification block")

    try:
        M = int(next(toks))
    except Exception:
        fail("missing M")
    if not (0 <= M <= PERIOD_CAP):
        fail("M out of range")

    order_lines = []
    try:
        for _ in range(M):
            t = int(next(toks))
            i = int(next(toks))
            q = int(next(toks))
            if not (1 <= t <= T):
                fail("order period out of range")
            if not (0 <= i < N):
                fail("order supplier index out of range")
            if not (0 <= q <= QTY_CAP):
                fail("order quantity out of range")
            order_lines.append((t, i, q))
    except SystemExit:
        raise
    except Exception:
        fail("bad order block")

    # reject any trailing garbage tokens? -- not required; extra whitespace is fine,
    # but stray non-numeric tokens would already have raised above via next()/int().
    # confirm no leftover unread *significant* tokens beyond what we consumed is not
    # required by the contract; we simply ignore anything after a well-formed block.

    F = evaluate_plan(inst, qual_actions, order_lines)
    B = internal_baseline(inst)
    B = max(B, 1e-6)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    sc = max(0.0, sc)
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

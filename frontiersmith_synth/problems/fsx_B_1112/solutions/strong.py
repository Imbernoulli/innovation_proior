# TIER: strong
# One shared preprocessing prefix, three different classical shortcuts:
#   SWEEP  -- the batch is an arithmetic progression, so seed only the first
#             n+1 points via Horner (n(n+1) mults total) and extend the rest of
#             the (much larger) batch with a Newton forward-difference table
#             walk that uses ONLY subtractions/additions -- free.
#   PROBE  -- the batch repeats only D=4 distinct relative offsets; evaluate
#             each DISTINCT value once via Horner and reuse the register for
#             every repeat (register reuse in the output section costs nothing).
#   AD-HOC -- genuinely unstructured points: plain Horner, no shortcut exists.
# The shared "prefix" is the Horner evaluator itself plus the register-DAG
# discipline that lets every later batch reuse whatever an earlier batch
# already computed; the real saving is recognising WHICH n+1 (resp. D) points
# actually need to be paid for, and reusing/extending them for free instead of
# re-deriving every one of the M query points independently.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    int(next(it))  # test_id (unused by this tier)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n + 1)]
    Q = int(next(it))
    q = [int(next(it)) for _ in range(Q)]
    m1 = int(next(it)); idx1 = [int(next(it)) for _ in range(m1)]
    m2 = int(next(it)); idx2 = [int(next(it)) for _ in range(m2)]
    m3 = int(next(it)); idx3 = [int(next(it)) for _ in range(m3)]

    instrs = []

    def emit(op, x, y):
        instrs.append("%s %s %s" % (op, x, y))
        return "r%d" % (len(instrs) - 1)

    def horner(qi):
        xreg = "q%d" % qi
        acc = "a%d" % n
        for k in range(n - 1, -1, -1):
            t = emit("mul", acc, xreg)
            acc = emit("add", t, "a%d" % k)
        return acc

    # ---- SWEEP batch: seed n+1 points, then finite-difference march ----
    seed_count = min(m1, n + 1)
    seeds = [horner(idx1[j]) for j in range(seed_count)]
    if seed_count <= n:
        # not enough points for the difference table to pay off -- just Horner
        sweep_outputs = seeds
    else:
        table = [seeds]
        for _r in range(1, n + 1):
            prev = table[-1]
            row = [emit("sub", prev[i + 1], prev[i]) for i in range(len(prev) - 1)]
            table.append(row)
        delta = [table[r][0] for r in range(n + 1)]
        sweep_outputs = [delta[0]]
        for _step in range(1, m1):
            new_delta = [None] * (n + 1)
            new_delta[n] = delta[n]
            for r in range(n - 1, -1, -1):
                new_delta[r] = emit("add", delta[r], delta[r + 1])
            delta = new_delta
            sweep_outputs.append(delta[0])

    # ---- PROBE batch: dedup repeated query values ----
    reg_for_val = {}
    probe_outputs = []
    for qi in idx2:
        if qi not in reg_for_val:
            reg_for_val[qi] = horner(qi)
        probe_outputs.append(reg_for_val[qi])

    # ---- AD-HOC batch: no shared structure, plain Horner ----
    adhoc_outputs = [horner(qi) for qi in idx3]

    outputs = sweep_outputs + probe_outputs + adhoc_outputs

    sys.stdout.write("%d\n" % len(instrs))
    sys.stdout.write("\n".join(instrs) + "\n")
    sys.stdout.write("%d\n" % len(outputs))
    sys.stdout.write("\n".join(r[1:] for r in outputs) + "\n")


if __name__ == "__main__":
    main()

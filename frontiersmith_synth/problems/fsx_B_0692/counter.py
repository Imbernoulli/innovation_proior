import sys

# Format D checker -- "Plate-Stack Codegen" (stack-shuffle-codegen family).
#
#   1) Parse the shared-value expression DAG (leaves + ADD/SUB/MUL nodes + K
#      designated outputs, in required emission order) from <in>.
#   2) Parse the participant's stack-machine program from <out>: one
#      instruction per non-empty line, chosen from
#         PUSH i | OP ADD/SUB/MUL | DUP | SWAP | OVER | ROT | STORE s | LOAD s | OUTPUT
#   3) EXACT-equality gate: simulate the program; every OUTPUT instruction must
#      pop, in order, exactly the DAG's true value (mod P) for out_1..out_K,
#      and every instruction must be feasible (no stack underflow, no read
#      from an uninitialized memory slot, no bad indices).  Any violation ->
#      Ratio: 0.0.
#   4) Objective (minimize) = total weighted instruction cost
#         PUSH/OP/DUP/SWAP/OVER/ROT/OUTPUT = 1,  STORE/LOAD = 8.
#      Baseline B = cost of the naive "recompute every subtree from scratch,
#      independently per output, no shuffles, no memory" construction.
#      Ratio = min(1, 0.1 * B / F).

MAX_INSTR = 200000
SLOT_LIMIT = 1000000
OPS = {"ADD", "SUB", "MUL"}


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_instance(text):
    toks = text.split()
    it = iter(toks)
    try:
        P = int(next(it))
        N = int(next(it)); M = int(next(it)); K = int(next(it))
        if not (1 <= N <= 200 and 1 <= M <= 2000 and 1 <= K <= 200):
            raise ValueError
        leaves = [int(next(it)) % P for _ in range(N)]
        nodes = {}
        for i in range(1, M + 1):
            op = next(it)
            cl = next(it); cr = next(it)
            if op not in OPS:
                raise ValueError
            nodes[i] = (op, cl, cr)
        outs = [next(it) for _ in range(K)]
    except (StopIteration, ValueError):
        raise ValueError("malformed instance")
    return P, N, M, K, leaves, nodes, outs


def refkey(tok):
    # "L3" -> ('L',3), "N17" -> ('N',17)
    if len(tok) < 2 or tok[0] not in ("L", "N"):
        raise ValueError("bad ref")
    return (tok[0], int(tok[1:]))


def compute_values_and_sizes(P, N, M, leaves, nodes):
    val = {}
    size = {}
    for i in range(1, N + 1):
        val[("L", i)] = leaves[i - 1]
        size[("L", i)] = 1
    for i in range(1, M + 1):
        op, cl, cr = nodes[i]
        clk, crk = refkey(cl), refkey(cr)
        a, b = val[clk], val[crk]
        if op == "ADD":
            v = (a + b) % P
        elif op == "SUB":
            v = (a - b) % P
        else:
            v = (a * b) % P
        val[("N", i)] = v
        size[("N", i)] = size[clk] + size[crk] + 1
    return val, size


def main():
    with open(sys.argv[1]) as f:
        inst_text = f.read()
    with open(sys.argv[2]) as f:
        out_text = f.read()

    try:
        P, N, M, K, leaves, nodes, outs = parse_instance(inst_text)
    except Exception:
        fail("checker could not parse its own instance (setter bug)")

    try:
        for cl, cr in ((refkey(c1), refkey(c2)) for _op, c1, c2 in nodes.values()):
            for k in (cl, cr):
                if k[0] == "L" and not (1 <= k[1] <= N):
                    raise ValueError
                if k[0] == "N" and not (1 <= k[1] <= M):
                    raise ValueError
        out_refs = [refkey(o) for o in outs]
        for k in out_refs:
            if k[0] == "L" and not (1 <= k[1] <= N):
                raise ValueError
            if k[0] == "N" and not (1 <= k[1] <= M):
                raise ValueError
    except Exception:
        fail("checker could not resolve its own DAG refs (setter bug)")

    val, size = compute_values_and_sizes(P, N, M, leaves, nodes)
    target_vals = [val[r] for r in out_refs]
    B = sum(size[r] for r in out_refs) + K
    if B <= 0:
        fail("degenerate baseline")

    # ---- parse participant program ----
    lines = [ln.strip() for ln in out_text.splitlines() if ln.strip() != ""]
    if not lines:
        fail("empty output")
    if len(lines) > MAX_INSTR:
        fail("instruction count too large")

    stack = []
    memory = {}
    stored = set()
    cost = 0
    out_idx = 0

    for lineno, line in enumerate(lines, 1):
        parts = line.split()
        opcode = parts[0]
        try:
            if opcode == "PUSH":
                if len(parts) != 2:
                    raise ValueError
                i = int(parts[1])
                if not (1 <= i <= N):
                    fail("PUSH out-of-range leaf index at line %d" % lineno)
                stack.append(leaves[i - 1])
                cost += 1
            elif opcode == "OP":
                if len(parts) != 2 or parts[1] not in OPS:
                    raise ValueError
                if len(stack) < 2:
                    fail("stack underflow on OP at line %d" % lineno)
                a = stack.pop()
                b = stack.pop()
                if parts[1] == "ADD":
                    v = (b + a) % P
                elif parts[1] == "SUB":
                    v = (b - a) % P
                else:
                    v = (b * a) % P
                stack.append(v)
                cost += 1
            elif opcode == "DUP":
                if len(parts) != 1:
                    raise ValueError
                if len(stack) < 1:
                    fail("stack underflow on DUP at line %d" % lineno)
                stack.append(stack[-1])
                cost += 1
            elif opcode == "SWAP":
                if len(parts) != 1:
                    raise ValueError
                if len(stack) < 2:
                    fail("stack underflow on SWAP at line %d" % lineno)
                stack[-1], stack[-2] = stack[-2], stack[-1]
                cost += 1
            elif opcode == "OVER":
                if len(parts) != 1:
                    raise ValueError
                if len(stack) < 2:
                    fail("stack underflow on OVER at line %d" % lineno)
                stack.append(stack[-2])
                cost += 1
            elif opcode == "ROT":
                if len(parts) != 1:
                    raise ValueError
                if len(stack) < 3:
                    fail("stack underflow on ROT at line %d" % lineno)
                v0, v1, v2 = stack[-1], stack[-2], stack[-3]
                stack[-3], stack[-2], stack[-1] = v1, v0, v2
                cost += 1
            elif opcode == "STORE":
                if len(parts) != 2:
                    raise ValueError
                s = int(parts[1])
                if not (0 <= s < SLOT_LIMIT):
                    fail("STORE slot out of range at line %d" % lineno)
                if len(stack) < 1:
                    fail("stack underflow on STORE at line %d" % lineno)
                memory[s] = stack.pop()
                stored.add(s)
                cost += 8
            elif opcode == "LOAD":
                if len(parts) != 2:
                    raise ValueError
                s = int(parts[1])
                if not (0 <= s < SLOT_LIMIT):
                    fail("LOAD slot out of range at line %d" % lineno)
                if s not in stored:
                    fail("LOAD from uninitialized slot %d at line %d" % (s, lineno))
                stack.append(memory[s])
                cost += 8
            elif opcode == "OUTPUT":
                if len(parts) != 1:
                    raise ValueError
                if len(stack) < 1:
                    fail("stack underflow on OUTPUT at line %d" % lineno)
                if out_idx >= K:
                    fail("more than K=%d OUTPUT instructions" % K)
                v = stack.pop()
                if v != target_vals[out_idx]:
                    fail("wrong value at output position %d (line %d)" % (out_idx + 1, lineno))
                out_idx += 1
                cost += 1
            else:
                raise ValueError
        except ValueError:
            fail("malformed instruction at line %d: %r" % (lineno, line))

    if out_idx != K:
        fail("only %d of K=%d outputs were produced" % (out_idx, K))

    sc = min(1000.0, 100.0 * B / max(1e-9, float(cost)))
    print("Plate-Stack Codegen: B=%d F=%d Ratio: %.6f" % (B, cost, sc / 1000.0))


if __name__ == "__main__":
    main()

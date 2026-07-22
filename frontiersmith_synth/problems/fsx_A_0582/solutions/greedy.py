# TIER: greedy
# The obvious first pass any engineer writes on a bloated program: a peephole
# optimiser -- constant folding, identity folds (x+0, x-0, x*1), copy propagation,
# and dead-code elimination.  It strips the dead scaffolding and the final copy,
# recovering the expanded-monomial core (~2x smaller).  But it is purely SYNTACTIC:
# it cannot re-derive the hidden recurrence, so it plateaus far above `strong`.
import sys

NIN = 8
P = 2147483647

def read_prog():
    it = iter(sys.stdin.read().split())
    p = int(next(it)); L = int(next(it))
    prog = []
    for _ in range(L):
        op = next(it)
        if op == "const":
            prog.append(("const", int(next(it)) % P, None))
        else:
            prog.append((op, int(next(it)), int(next(it))))
    return prog

def main():
    prog = read_prog()
    nvals = NIN + len(prog)
    is_const = [False] * nvals
    cval = [0] * nvals
    rep = list(range(nvals))               # copy-propagation union-find (chain)

    def R(i):
        while rep[i] != i:
            i = rep[i]
        return i

    newins = [None] * nvals
    for t, ins in enumerate(prog):
        idx = NIN + t
        if ins[0] == "const":
            is_const[idx] = True; cval[idx] = ins[1] % P
            newins[idx] = ("const", ins[1] % P); continue
        a = R(ins[1]); b = R(ins[2]); op = ins[0]
        if is_const[a] and is_const[b]:
            va, vb = cval[a], cval[b]
            r = (va + vb) % P if op == "add" else (va - vb) % P if op == "sub" else (va * vb) % P
            is_const[idx] = True; cval[idx] = r; newins[idx] = ("const", r); continue
        if op == "add" and is_const[a] and cval[a] == 0: rep[idx] = b; newins[idx] = ("copy", b); continue
        if op == "add" and is_const[b] and cval[b] == 0: rep[idx] = a; newins[idx] = ("copy", a); continue
        if op == "sub" and is_const[b] and cval[b] == 0: rep[idx] = a; newins[idx] = ("copy", a); continue
        if op == "mul" and is_const[b] and cval[b] == 1 % P: rep[idx] = a; newins[idx] = ("copy", a); continue
        if op == "mul" and is_const[a] and cval[a] == 1 % P: rep[idx] = b; newins[idx] = ("copy", b); continue
        newins[idx] = (op, a, b)

    result = R(NIN + len(prog) - 1)

    live = set(); stack = [result]
    while stack:
        x = stack.pop()
        if x < NIN or x in live:
            continue
        live.add(x)
        ins = newins[x]
        if ins[0] == "const":
            pass
        elif ins[0] == "copy":
            stack.append(ins[1])
        else:
            stack.append(ins[1]); stack.append(ins[2])

    remap = {i: i for i in range(NIN)}
    out = []
    for i in sorted(v for v in live if v >= NIN):
        ins = newins[i]
        if ins[0] == "copy":
            src = R(ins[1])
            remap[i] = remap[src] if src < NIN else remap[src]
            continue
        if ins[0] == "const":
            out.append("const %d" % ins[1]); remap[i] = NIN + len(out) - 1
        else:
            out.append("%s %d %d" % (ins[0], remap[ins[1]], remap[ins[2]]))
            remap[i] = NIN + len(out) - 1

    res_new = remap[result] if result >= NIN else result
    if not out or (NIN + len(out) - 1) != res_new:
        out.append("const 0"); z = NIN + len(out) - 1
        out.append("add %d %d" % (res_new, z))

    sys.stdout.write("\n".join([str(len(out))] + out) + "\n")

if __name__ == "__main__":
    main()

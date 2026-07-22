import sys, re, random

# Format D checker -- macro-library-compression.
#
# <in>  : a flat corpus of N straight-line programs (SLPs) over M shared input
#         variables x0..x{M-1}, modulus P (documented), seed SEED.
# <out> : participant artifact =
#           MACROS K
#           K macro defs: MACRO name arity B / B body lines / RET t<idx>
#           PROGRAMS N
#           N rewritten programs: PROGRAM k L' / L' lines (OP a b | CALL name args...) / OUT t<idx>
#
# Feasibility: strict schema/range validation, THEN exact functional equivalence
# -- each rewritten program must compute the identical value (mod P) as the
# original on S independently random assignments of x0..x{M-1} (Schwartz-Zippel
# style: our programs have tiny total degree so a handful of random points over
# a large prime field give equivalence with overwhelming probability).
#
# Objective (minimize): F = sum of rewritten program lengths (each OP or CALL
# line costs 1) + 3 * (sum of body sizes of macros that are actually CALLed
# anywhere) -- i.e. amortized library cost: pay the definition once, call cheaply.
# Baseline B = sum of ORIGINAL (unrewritten) program lengths.  ratio = min(1, 0.1*B/F).

MAXOUT_BYTES = 3_000_000
MAX_MACROS = 8
MAX_ARITY = 6
MAX_BODY = 12
MAX_CONST = 1000
MAX_PROG_LEN = 2000
S_POINTS = 6

X_RE = re.compile(r"^x(0|[1-9]\d*)$")
C_RE = re.compile(r"^c(-?(0|[1-9]\d*))$")
T_RE = re.compile(r"^t(0|[1-9]\d*)$")
P_RE = re.compile(r"^p(0|[1-9]\d*)$")
OPS = {"ADD", "SUB", "MUL"}


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


class Tok:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0
        self.n = len(toks)

    def next(self):
        if self.i >= self.n:
            raise IndexError("unexpected end of tokens")
        t = self.toks[self.i]
        self.i += 1
        return t

    def next_int(self):
        return int(self.next())


def parse_instance(text):
    toks = text.split()
    it = Tok(toks)
    N = it.next_int(); M = it.next_int(); P = it.next_int(); SEED = it.next_int()
    progs = []
    for k in range(N):
        assert it.next() == "PROGRAM"
        kk = it.next_int(); L = it.next_int()
        assert kk == k
        instr = []
        for i in range(L):
            op = it.next(); a = it.next(); b = it.next()
            instr.append((op, a, b))
        assert it.next() == "OUT"
        out_tok = it.next()
        m = T_RE.match(out_tok)
        out_idx = int(m.group(1))
        progs.append((instr, out_idx))
    return N, M, P, SEED, progs


def eval_value(tok, M, xvals, temps):
    m = X_RE.match(tok)
    if m:
        idx = int(m.group(1))
        if idx >= M:
            raise ValueError("x out of range")
        return xvals[idx]
    m = C_RE.match(tok)
    if m:
        return int(m.group(1))
    m = T_RE.match(tok)
    if m:
        idx = int(m.group(1))
        if idx >= len(temps):
            raise ValueError("t use-before-def")
        return temps[idx]
    raise ValueError("bad operand token %r" % tok)


def run_program(instr, out_idx, M, P, xvals):
    temps = []
    for (op, a, b) in instr:
        va = eval_value(a, M, xvals, temps)
        vb = eval_value(b, M, xvals, temps)
        if op == "ADD":
            v = (va + vb) % P
        elif op == "SUB":
            v = (va - vb) % P
        elif op == "MUL":
            v = (va * vb) % P
        else:
            raise ValueError("bad op")
        temps.append(v)
    if out_idx >= len(temps):
        raise ValueError("OUT out of range")
    return temps[out_idx]


def main():
    inst_text = open(sys.argv[1]).read()
    try:
        N, M, P, SEED, progs = parse_instance(inst_text)
    except Exception as e:
        fail("bad instance (should not happen): %s" % e)

    B = sum(len(instr) for (instr, _) in progs)
    if B <= 0:
        fail("degenerate zero-cost instance")

    try:
        out_bytes = open(sys.argv[2], "rb").read()
    except Exception:
        fail("cannot read output")
    if len(out_bytes) == 0:
        fail("empty output")
    if len(out_bytes) > MAXOUT_BYTES:
        fail("output too large")
    try:
        out_text = out_bytes.decode("utf-8", errors="strict")
    except Exception:
        fail("output not valid utf-8")

    toks = out_text.split()
    it = Tok(toks)

    # ---- parse MACROS section ----
    try:
        if it.next() != "MACROS":
            fail("missing MACROS header")
        K = it.next_int()
    except SystemExit:
        raise
    except Exception:
        fail("bad MACROS header")
    if not (0 <= K <= MAX_MACROS):
        fail("macro count out of range")

    macros = {}   # name -> dict(arity, body=[(op,a,b)], ret)
    order = []
    try:
        for _ in range(K):
            if it.next() != "MACRO":
                fail("missing MACRO header")
            name = it.next()
            arity = it.next_int()
            bsz = it.next_int()
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", name):
                fail("bad macro name")
            if name in macros:
                fail("duplicate macro name")
            if not (0 <= arity <= MAX_ARITY):
                fail("bad macro arity")
            if not (1 <= bsz <= MAX_BODY):
                fail("bad macro body size")
            body = []
            for i in range(bsz):
                op = it.next(); a = it.next(); b = it.next()
                if op not in OPS:
                    fail("bad macro op")
                for tok in (a, b):
                    mm = P_RE.match(tok)
                    if mm:
                        if int(mm.group(1)) >= arity:
                            fail("macro param out of range")
                        continue
                    mm = T_RE.match(tok)
                    if mm:
                        if int(mm.group(1)) >= i:
                            fail("macro body use-before-def")
                        continue
                    mm = C_RE.match(tok)
                    if mm:
                        if abs(int(mm.group(1))) > MAX_CONST:
                            fail("macro constant too large")
                        continue
                    mm = X_RE.match(tok)
                    if mm:
                        if int(mm.group(1)) >= M:
                            fail("macro free-var out of range")
                        continue
                    fail("bad macro operand token %r" % tok)
                body.append((op, a, b))
            if it.next() != "RET":
                fail("missing RET")
            ret_tok = it.next()
            mm = T_RE.match(ret_tok)
            if not mm or int(mm.group(1)) >= bsz:
                fail("bad RET target")
            ret_idx = int(mm.group(1))
            macros[name] = dict(arity=arity, body=body, ret=ret_idx)
            order.append(name)
    except SystemExit:
        raise
    except Exception as e:
        fail("malformed macro section: %s" % e)

    # ---- parse PROGRAMS section ----
    try:
        if it.next() != "PROGRAMS":
            fail("missing PROGRAMS header")
        Nout = it.next_int()
    except SystemExit:
        raise
    except Exception:
        fail("bad PROGRAMS header")
    if Nout != N:
        fail("program count mismatch (got %d need %d)" % (Nout, N))

    rew_progs = []
    used_macro_calls = {}
    try:
        for k in range(N):
            if it.next() != "PROGRAM":
                fail("missing PROGRAM header")
            kk = it.next_int(); Lp = it.next_int()
            if kk != k:
                fail("program index out of order")
            if not (1 <= Lp <= MAX_PROG_LEN):
                fail("bad rewritten program length")
            rinstr = []
            for i in range(Lp):
                head = it.next()
                if head == "CALL":
                    name = it.next()
                    if name not in macros:
                        fail("CALL to undefined macro %r" % name)
                    arity = macros[name]["arity"]
                    args = [it.next() for _ in range(arity)]
                    for tok in args:
                        ok = False
                        for RX in (X_RE, C_RE, T_RE):
                            mm = RX.match(tok)
                            if mm:
                                ok = True
                                if RX is X_RE and int(mm.group(1)) >= M:
                                    fail("call arg x out of range")
                                if RX is T_RE and int(mm.group(1)) >= i:
                                    fail("call arg use-before-def")
                                if RX is C_RE and abs(int(mm.group(1))) > MAX_CONST:
                                    fail("call arg constant too large")
                                break
                        if not ok:
                            fail("bad call arg token %r" % tok)
                    rinstr.append(("CALL", name, args))
                    used_macro_calls[name] = used_macro_calls.get(name, 0) + 1
                elif head in OPS:
                    a = it.next(); b = it.next()
                    for tok in (a, b):
                        ok = False
                        for RX in (X_RE, C_RE, T_RE):
                            mm = RX.match(tok)
                            if mm:
                                ok = True
                                if RX is X_RE and int(mm.group(1)) >= M:
                                    fail("op arg x out of range")
                                if RX is T_RE and int(mm.group(1)) >= i:
                                    fail("op arg use-before-def")
                                if RX is C_RE and abs(int(mm.group(1))) > MAX_CONST:
                                    fail("op arg constant too large")
                                break
                        if not ok:
                            fail("bad op arg token %r" % tok)
                    rinstr.append((head, a, b))
                else:
                    fail("bad instruction head %r" % head)
            if it.next() != "OUT":
                fail("missing OUT")
            out_tok = it.next()
            mm = T_RE.match(out_tok)
            if not mm or int(mm.group(1)) >= Lp:
                fail("bad OUT target")
            rew_progs.append((rinstr, int(mm.group(1))))
    except SystemExit:
        raise
    except Exception as e:
        fail("malformed program section: %s" % e)

    # ---- equivalence check on S random field points (shared X per point) ----
    rng = random.Random(0x5EED0000 ^ (SEED * 2654435761 + 97))

    def eval_rewritten(instr, out_idx, xvals):
        temps = []
        for i, ins in enumerate(instr):
            if ins[0] == "CALL":
                _, name, args = ins
                argvals = [eval_value(a, M, xvals, temps) for a in args]
                v = eval_macro(macros[name], argvals, xvals)
            else:
                op, a, b = ins
                va = eval_value(a, M, xvals, temps)
                vb = eval_value(b, M, xvals, temps)
                if op == "ADD":
                    v = (va + vb) % P
                elif op == "SUB":
                    v = (va - vb) % P
                else:
                    v = (va * vb) % P
            temps.append(v)
        return temps[out_idx]

    def eval_macro(mdef, argvals, xvals):
        temps = []
        for (op, a, b) in mdef["body"]:
            va = macro_operand(a, argvals, xvals, temps)
            vb = macro_operand(b, argvals, xvals, temps)
            if op == "ADD":
                v = (va + vb) % P
            elif op == "SUB":
                v = (va - vb) % P
            else:
                v = (va * vb) % P
            temps.append(v)
        return temps[mdef["ret"]]

    def macro_operand(tok, argvals, xvals, temps):
        m = P_RE.match(tok)
        if m:
            return argvals[int(m.group(1))]
        m = C_RE.match(tok)
        if m:
            return int(m.group(1))
        m = T_RE.match(tok)
        if m:
            return temps[int(m.group(1))]
        m = X_RE.match(tok)
        if m:
            return xvals[int(m.group(1))]
        raise ValueError("bad macro operand")

    for s in range(S_POINTS):
        xvals = [rng.randrange(P) for _ in range(M)]
        for k in range(N):
            orig_instr, orig_out = progs[k]
            try:
                orig_val = run_program(orig_instr, orig_out, M, P, xvals)
            except Exception as e:
                fail("internal error evaluating original program %d: %s" % (k, e))
            rinstr, rout = rew_progs[k]
            try:
                new_val = eval_rewritten(rinstr, rout, xvals)
            except Exception as e:
                fail("runtime error evaluating rewritten program %d: %s" % (k, e))
            if orig_val != new_val:
                fail("output mismatch on program %d (point %d)" % (k, s))

    # ---- cost ----
    F = sum(len(rinstr) for (rinstr, _) in rew_progs)
    F += 3 * sum(len(macros[name]["body"]) for name in used_macro_calls)

    if F <= 0:
        fail("degenerate zero cost")

    ratio = min(1.0, 0.1 * B / F)
    print("B=%d F=%d macros_used=%d Ratio: %.6f" % (B, F, len(used_macro_calls), ratio))


if __name__ == "__main__":
    main()

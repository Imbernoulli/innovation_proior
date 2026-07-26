#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic op-count scorer for the
cheap-spectral-projector (high-tone sieve) problem.

The participant emits a straight-line vector program over a fixed register
file (register 0 = probe vector, replayed once per probe). Instructions:
  NVEC m
  MATVEC d s            reg[d] = A @ reg[s]
  AXPBY  d a s1 b s2     reg[d] = a*reg[s1] + b*reg[s2]
  SCALE  d a s           reg[d] = a*reg[s]
  COPY   d s             reg[d] = reg[s]
  CSOLVE d a b s         reg[d] = Re[ (A - (a+bi)I)^-1 @ reg[s] ]
  OUTPUT s               (exactly one, last line)

We FIRST verify accuracy against the exact spectral projector (via eigh,
independent of the participant's method), THEN count exact scalar ops.
"""
import sys
import math
import numpy as np

MAX_NVEC = 64
MAX_INSTR = 20000
D_TRIV = 350       # must match solutions/trivial.py's fixed degree
K_B = 6.0          # baseline-formula constant


def fail(reason):
    print("Infeasible: %s Ratio: 0.0" % reason)
    sys.exit(0)


def parse_in(path):
    with open(path) as f:
        toks = f.read().split("\n")
    toks = [t for t in toks if t.strip() != ""]
    idx = 0
    n = int(toks[idx].split()[0]); idx += 1
    theta, eps, b_bw = toks[idx].split(); idx += 1
    theta = float(theta); eps = float(eps); b_bw = int(b_bw)
    gap = float(toks[idx].split()[0]); idx += 1
    nnz = int(toks[idx].split()[0]); idx += 1
    A = np.zeros((n, n))
    for _ in range(nnz):
        i, j, v = toks[idx].split(); idx += 1
        i = int(i); j = int(j); v = float(v)
        A[i, j] = v
        A[j, i] = v
    k = int(toks[idx].split()[0]); idx += 1
    probes = []
    for _ in range(k):
        row = [float(x) for x in toks[idx].split()]; idx += 1
        probes.append(np.array(row))
    nnz_full = n + 2 * (nnz - n)  # off-diagonal entries mirrored
    return dict(n=n, theta=theta, eps=eps, b_bw=b_bw, gap=gap, A=A,
                probes=probes, nnz_full=nnz_full)


def parse_program(path, nmax):
    with open(path) as f:
        raw = f.read()
    lines = [ln.strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        fail("empty output")
    if len(lines) - 1 > MAX_INSTR:
        fail("too many instructions")

    def parse_float(tok):
        low = tok.lower().lstrip("+-")
        if low in ("nan", "inf", "infinity"):
            fail("non-finite token")
        try:
            v = float(tok)
        except ValueError:
            fail("bad float token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite value")
        return v

    def parse_reg(tok, nvec):
        try:
            r = int(tok)
        except ValueError:
            fail("bad register token %r" % tok)
        if r < 0 or r >= nvec:
            fail("register out of range")
        return r

    head = lines[0].split()
    if len(head) != 2 or head[0] != "NVEC":
        fail("first line must be NVEC m")
    try:
        nvec = int(head[1])
    except ValueError:
        fail("bad NVEC value")
    if nvec < 1 or nvec > MAX_NVEC:
        fail("NVEC out of range")

    prog = []
    output_reg = None
    for li, ln in enumerate(lines[1:]):
        parts = ln.split()
        op = parts[0]
        if op == "OUTPUT":
            if li != len(lines) - 2:
                fail("OUTPUT must be the last instruction")
            if len(parts) != 2:
                fail("bad OUTPUT arity")
            output_reg = parse_reg(parts[1], nvec)
        elif op == "MATVEC":
            if len(parts) != 3:
                fail("bad MATVEC arity")
            d = parse_reg(parts[1], nvec); s = parse_reg(parts[2], nvec)
            if d == 0:
                fail("cannot write register 0")
            prog.append(("MATVEC", d, s))
        elif op == "AXPBY":
            if len(parts) != 6:
                fail("bad AXPBY arity")
            d = parse_reg(parts[1], nvec)
            a = parse_float(parts[2])
            s1 = parse_reg(parts[3], nvec)
            b = parse_float(parts[4])
            s2 = parse_reg(parts[5], nvec)
            if d == 0:
                fail("cannot write register 0")
            prog.append(("AXPBY", d, a, s1, b, s2))
        elif op == "SCALE":
            if len(parts) != 4:
                fail("bad SCALE arity")
            d = parse_reg(parts[1], nvec)
            a = parse_float(parts[2])
            s = parse_reg(parts[3], nvec)
            if d == 0:
                fail("cannot write register 0")
            prog.append(("SCALE", d, a, s))
        elif op == "COPY":
            if len(parts) != 3:
                fail("bad COPY arity")
            d = parse_reg(parts[1], nvec); s = parse_reg(parts[2], nvec)
            if d == 0:
                fail("cannot write register 0")
            prog.append(("COPY", d, s))
        elif op == "CSOLVE":
            if len(parts) != 5:
                fail("bad CSOLVE arity")
            d = parse_reg(parts[1], nvec)
            a = parse_float(parts[2])
            b = parse_float(parts[3])
            s = parse_reg(parts[4], nvec)
            if d == 0:
                fail("cannot write register 0")
            prog.append(("CSOLVE", d, a, b, s))
        else:
            fail("unknown opcode %r" % op)
    if output_reg is None:
        fail("missing OUTPUT")
    return nvec, prog, output_reg


def compute_cost(prog, nnz_full, n, b_bw):
    cost = 0
    for instr in prog:
        op = instr[0]
        if op == "MATVEC":
            cost += 2 * nnz_full
        elif op == "AXPBY":
            cost += 3 * n
        elif op == "SCALE":
            cost += n
        elif op == "COPY":
            cost += 0
        elif op == "CSOLVE":
            cost += 8 * n * b_bw * b_bw + 4 * n * b_bw
    return cost


def run_program(prog, output_reg, nvec, A, v0, n):
    regs = [None] * nvec
    regs[0] = v0.astype(float).copy()
    for r in range(1, nvec):
        regs[r] = np.zeros(n)
    ident = np.eye(n)
    for instr in prog:
        op = instr[0]
        if op == "MATVEC":
            _, d, s = instr
            regs[d] = A @ regs[s]
        elif op == "AXPBY":
            _, d, a, s1, b, s2 = instr
            regs[d] = a * regs[s1] + b * regs[s2]
        elif op == "SCALE":
            _, d, a, s = instr
            regs[d] = a * regs[s]
        elif op == "COPY":
            _, d, s = instr
            regs[d] = regs[s].copy()
        elif op == "CSOLVE":
            _, d, a, b, s = instr
            shift = complex(a, b)
            M = A - shift * ident
            rhs = regs[s].astype(complex)
            try:
                sol = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                fail("singular CSOLVE shift")
            regs[d] = sol.real
        if not np.all(np.isfinite(regs[instr[1]])):
            fail("non-finite intermediate value")
    out = regs[output_reg]
    if not np.all(np.isfinite(out)):
        fail("non-finite output")
    return out


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = parse_in(in_path)
    n, theta, eps, gap = inst["n"], inst["theta"], inst["eps"], inst["gap"]
    A = inst["A"]; probes = inst["probes"]; nnz_full = inst["nnz_full"]
    b_bw = inst["b_bw"]

    nvec, prog, output_reg = parse_program(out_path, n)

    # exact ground truth via eigendecomposition (independent oracle)
    Lam, Q = np.linalg.eigh(A)
    hi_mask = (Lam > theta).astype(float)

    for v in probes:
        y = run_program(prog, output_reg, nvec, A, v, n)
        yref = Q @ (hi_mask * (Q.T @ v))
        err = float(np.linalg.norm(y - yref))
        if err > eps:
            fail("accuracy %.6g exceeds epsilon %.6g" % (err, eps))

    F = compute_cost(prog, nnz_full, n, b_bw)
    if F <= 0:
        fail("zero-cost program cannot be a valid filter")

    unit = 2 * nnz_full + 9 * n
    B = math.ceil(K_B / gap) * unit
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("OK F=%d B=%d Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()

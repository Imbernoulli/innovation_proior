#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for the editable-
archive chunking problem (Format D, eval_form=flops).

Pipeline:
  1. Read the instance: base corpus, splice-in edit, cost weights, queries.
  2. Parse the participant's straight-line CHUNKING PROGRAM under a strict,
     bounded register-machine schema (fixed opcode set, <=40 instructions,
     single final RESULT). Any structural violation -> Ratio: 0.0.
  3. Re-execute the program once per byte position over the base corpus
     (fresh memory) to get block boundaries, and once more (fresh memory,
     same program) over the EDITED corpus. This "re-execute on the stated
     inputs" step stands in for "verify equivalence" in this family: the
     boundaries are exactly whatever this deterministic program computes,
     nothing else.
  4. Compute F = WC*CompressedSize + WI*IndexCost + WS*SeekCost + WE*EditCost
     from the two boundary sets (all exact integer arithmetic). Internal
     baseline B = the same formula for the legal "never cut" (one giant
     block) construction. Minimization: Ratio = min(1.0, 0.1*B/max(1e-9,F)).
Nothing is timed and there is no randomness.
"""
import sys

REG_N = 20
MAX_OPS = 40
MAX_IMM = 1_000_000
CAP = 10 ** 9
MEM_N = 8          # persistent memory slots m0..m7
RESULT_ARITY = 1 + MEM_N   # c, m0..m7

OPCODES = {
    'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2, 'MIN': 2, 'MAX': 2, 'LT': 2,
    'SEL': 3, 'MOV': 1,
}

HEADER_BASE = 6
DEDUP_REF_BYTES = 2
INDEX_ENTRY_BYTES = 4
SEEK_BYTE_COST = 1
ECOST_UNMATCHED = 1


# ---------------------------------------------------------------- parsing --
def parse_reg(tok):
    if not tok.startswith('r'):
        return None
    rest = tok[1:]
    if rest.isdigit():
        idx = int(rest)
        if 0 <= idx < REG_N:
            return idx
    return None


def parse_operand(tok):
    """Register ref 'rN', or a bare (optionally signed) integer literal --
    deliberately NOT '#'-prefixed so that an adversarial 'nan'/'inf'
    substitution into a numeric slot fails isdigit() and is rejected outright,
    instead of being an inert no-op the checker never has to reject."""
    if tok.startswith('r'):
        idx = parse_reg(tok)
        return ('r', idx) if idx is not None else None
    body = tok
    sign = 1
    if body.startswith('-'):
        sign = -1
        body = body[1:]
    if body != '' and body.isdigit():
        v = sign * int(body)
        if -MAX_IMM <= v <= MAX_IMM:
            return ('i', v)
    return None


def parse_program(text):
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s[0] in '#;':
            continue
        if len(s) > 200:
            return False, "line too long", None, None
        lines.append(s)
    if len(lines) > MAX_OPS + 1:
        return False, f"too many non-blank lines ({len(lines)})", None, None
    if not lines:
        return False, "empty program", None, None
    for ln in lines[:-1]:
        if ln.split()[0] == 'RESULT':
            return False, "RESULT must be the single, final line", None, None
    if lines[-1].split()[0] != 'RESULT':
        return False, "missing RESULT as final line", None, None
    body = lines[:-1]
    if len(body) > MAX_OPS:
        return False, f"op count {len(body)} exceeds budget {MAX_OPS}", None, None
    instrs = []
    for ln in body:
        toks = ln.split()
        op = toks[0]
        if op not in OPCODES:
            return False, f"unknown opcode: {op}", None, None
        nsrc = OPCODES[op]
        if len(toks) != 2 + nsrc:
            return False, f"bad arity for {op}: '{ln}'", None, None
        dst = parse_reg(toks[1])
        if dst is None:
            return False, f"bad dst register: '{ln}'", None, None
        srcs = []
        for t in toks[2:]:
            o = parse_operand(t)
            if o is None:
                return False, f"bad operand '{t}' in: '{ln}'", None, None
            srcs.append(o)
        instrs.append((op, dst, srcs))
    rtoks = lines[-1].split()
    if len(rtoks) != 1 + RESULT_ARITY:
        return False, f"RESULT needs exactly {RESULT_ARITY} operands: c m0..m{MEM_N-1}", None, None
    result_ops = []
    for t in rtoks[1:]:
        o = parse_operand(t)
        if o is None:
            return False, f"bad RESULT operand: '{t}'", None, None
        result_ops.append(o)
    return True, "", instrs, result_ops


def resolve(o, regs):
    return regs[o[1]] if o[0] == 'r' else o[1]


def run_program(instrs, regs):
    for op, dst, srcs in instrs:
        vals = [resolve(s, regs) for s in srcs]
        if op == 'ADD':
            v = vals[0] + vals[1]
        elif op == 'SUB':
            v = vals[0] - vals[1]
        elif op == 'MUL':
            v = vals[0] * vals[1]
        elif op == 'DIV':
            v = 0 if vals[1] == 0 else vals[0] // vals[1]
        elif op == 'MIN':
            v = vals[0] if vals[0] < vals[1] else vals[1]
        elif op == 'MAX':
            v = vals[0] if vals[0] > vals[1] else vals[1]
        elif op == 'LT':
            v = 1 if vals[0] < vals[1] else 0
        elif op == 'SEL':
            v = vals[1] if vals[0] != 0 else vals[2]
        elif op == 'MOV':
            v = vals[0]
        else:
            raise RuntimeError("unreachable opcode")
        if v > CAP:
            v = CAP
        elif v < -CAP:
            v = -CAP
        regs[dst] = v


# ------------------------------------------------------------- instance io --
def read_instance(path):
    toks = open(path).read().split()
    i = 0

    def nxt():
        nonlocal i
        v = toks[i]
        i += 1
        return v

    N = int(nxt()); K = int(nxt())
    corpus = [int(nxt()) for _ in range(N)]
    ins_pos = int(nxt()); ins_len = int(nxt())
    ins_vals = [int(nxt()) for _ in range(ins_len)]
    WC = int(nxt()); WI = int(nxt()); WS = int(nxt()); WE = int(nxt())
    Q = int(nxt())
    queries = [int(nxt()) for _ in range(Q)]
    return dict(N=N, K=K, corpus=corpus, ins_pos=ins_pos, ins_vals=ins_vals,
                WC=WC, WI=WI, WS=WS, WE=WE, queries=queries)


# ------------------------------------------------------------- simulation --
def run_chunker(instrs, result_ops, seq, K):
    """Executes the program once per position of seq (fresh memory at the
    start), returns the sorted list of block-start offsets (always incl. 0)."""
    mem = [0] * MEM_N
    bounds = [0]
    for t, byte in enumerate(seq):
        regs = [0] * REG_N
        regs[0:MEM_N] = mem
        regs[8] = byte
        regs[9] = t
        regs[10] = len(seq)
        regs[11] = K
        work = regs[:]
        run_program(instrs, work)
        c = resolve(result_ops[0], work)
        newmem = [resolve(result_ops[1 + k], work) for k in range(MEM_N)]
        if c != c or c in (float('inf'), float('-inf')):
            c = 0
        if t > 0 and c != 0:
            bounds.append(t)
        mem = newmem
    return bounds


def blocks_from_bounds(seq, bounds):
    blocks = []
    for i in range(len(bounds)):
        s = bounds[i]
        e = bounds[i + 1] if i + 1 < len(bounds) else len(seq)
        blocks.append(tuple(seq[s:e]))
    return blocks


def compressed_size(blocks, K):
    bits = max(1, (K - 1).bit_length())
    total = 0
    seen = set()
    for b in blocks:
        if b in seen:
            total += DEDUP_REF_BYTES
        else:
            seen.add(b)
            payload = (len(b) * bits + 7) // 8
            total += HEADER_BASE + payload
    return total


def index_cost(blocks):
    return INDEX_ENTRY_BYTES * len(blocks)


def seek_cost(bounds, queries):
    if not queries:
        return 0
    total = 0
    for q in queries:
        lo, hi, blk = 0, len(bounds) - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if bounds[mid] <= q:
                blk = mid; lo = mid + 1
            else:
                hi = mid - 1
        total += (q - bounds[blk] + 1)
    return SEEK_BYTE_COST * total


def edit_cost(base_blocks, edited_blocks):
    base_set = set(base_blocks)
    total = 0
    for b in edited_blocks:
        if b not in base_set:
            total += len(b)
    return ECOST_UNMATCHED * total


def evaluate(instrs, result_ops, inst):
    corpus = inst['corpus']
    K = inst['K']
    bounds = run_chunker(instrs, result_ops, corpus, K)
    blocks = blocks_from_bounds(corpus, bounds)
    cs = compressed_size(blocks, K)
    ic = index_cost(blocks)
    sc = seek_cost(bounds, inst['queries'])
    edited = corpus[:inst['ins_pos']] + inst['ins_vals'] + corpus[inst['ins_pos']:]
    ebounds = run_chunker(instrs, result_ops, edited, K)
    eblocks = blocks_from_bounds(edited, ebounds)
    ec = edit_cost(blocks, eblocks)
    F = inst['WC'] * cs + inst['WI'] * ic + inst['WS'] * sc + inst['WE'] * ec
    return F, dict(cs=cs, ic=ic, sc=sc, ec=ec, nblocks=len(blocks))


NEVER_CUT_TEXT = "RESULT 0 0 0 0 0 0 0 0 0"


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = read_instance(in_path)

    try:
        prog_text = open(out_path, 'r', errors='replace').read()
    except Exception:
        print("INVALID: cannot read output")
        print("Ratio: 0.0")
        return

    if len(prog_text) > 20000:
        print("INVALID: output too large")
        print("Ratio: 0.0")
        return

    ok, reason, instrs, result_ops = parse_program(prog_text)
    if not ok:
        print(f"INVALID: {reason}")
        print("Ratio: 0.0")
        return

    F, comp = evaluate(instrs, result_ops, inst)

    if F != F or F in (float('inf'), float('-inf')) or F < 0:
        print("INVALID: non-finite or negative cost")
        print("Ratio: 0.0")
        return

    ok_b, _, b_instrs, b_result_ops = parse_program(NEVER_CUT_TEXT)
    B, _ = evaluate(b_instrs, b_result_ops, inst)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d cs=%d ic=%d sc=%d ec=%d nblocks=%d" %
          (F, B, comp['cs'], comp['ic'], comp['sc'], comp['ec'], comp['nblocks']))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

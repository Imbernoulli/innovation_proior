#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic scorer for the shared-link
congestion-controller problem (Format D, eval_form=flops).

Pipeline:
  1. Read the instance (link + competing-flow schedule).
  2. Parse the participant's straight-line window-update PROGRAM under a strict,
     bounded register-machine schema (fixed opcode set, backward-only dataflow
     within a tick, <=40 instructions). Any structural violation -> Ratio: 0.0.
  3. Re-execute the program once per simulated tick inside a deterministic,
     integer-only link simulator (shared FIFO queue, proportional drop-tail,
     scripted competing flows) to get the SAME artifact's real behaviour --
     this stands in for "verify exact equivalence" in this family: the program
     must be a well-formed deterministic function of the stated signals.
  4. Compute cost F = ALPHA*(1-Jain) + BETA*avg_queue_occupancy + GAMMA*underutilization.
     Internal baseline B = the same simulator run with a NEVER-ADAPT ego flow
     (constant initial window forever) -- a legitimate feasible construction.
     Minimization: Ratio = min(1.0, 0.1 * B / max(1e-9, F)).
Everything is exact integer arithmetic except the final scalar combination;
nothing is timed and there is no randomness.
"""
import sys

REG_N = 20            # registers 0..19
MAX_OPS = 40           # max non-RESULT instruction lines
MAX_IMM = 1_000_000    # max |immediate literal|
CAP = 10 ** 9          # per-register clamp (prevents runaway bignum blowup)
CWND_CAP = 100_000      # clamp on the emitted window itself

OPCODES = {  # opcode -> number of SOURCE operands (dst is separate)
    'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2, 'MIN': 2, 'MAX': 2, 'LT': 2,
    'SEL': 3, 'MOV': 1,
}


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
    deliberately NOT '#'-prefixed so that 'nan'/'inf' substituted into a
    numeric slot fails isdigit() and is rejected, rather than being an inert
    no-op substitution the checker never actually has to reject."""
    if tok.startswith('r') and len(tok) > 1 and tok[1:].isdigit():
        idx = int(tok[1:])
        if 0 <= idx < REG_N:
            return ('r', idx)
        return None
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
    """Returns (ok, reason, instrs, result_ops)."""
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
    if len(rtoks) != 6:
        return False, "RESULT needs exactly 5 operands: c m0 m1 m2 m3", None, None
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
    """Executes instrs in place on regs (length REG_N), clamping every write
    to [-CAP, CAP] so a pathological chain (e.g. repeated squaring) cannot
    blow up into an astronomically large integer."""
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

    T = int(nxt()); C = int(nxt()); Qmax = int(nxt()); n_comp = int(nxt())
    base_rtt_ego = int(nxt()); init_cwnd = int(nxt())
    ALPHA = float(nxt()); BETA = float(nxt()); GAMMA = float(nxt())
    comps = []
    for _ in range(n_comp):
        typ = nxt()
        p1 = int(nxt()); p2 = int(nxt()); p3 = int(nxt())
        brtt = int(nxt()); demand = int(nxt())
        comps.append({'type': typ, 'p1': p1, 'p2': p2, 'p3': p3,
                      'base_rtt': brtt, 'demand': demand})
    return {'T': T, 'C': C, 'Qmax': Qmax, 'n_comp': n_comp,
            'base_rtt_ego': base_rtt_ego, 'init_cwnd': init_cwnd,
            'ALPHA': ALPHA, 'BETA': BETA, 'GAMMA': GAMMA, 'comps': comps}


def allocate_drops(arrivals, drop_total):
    """Deterministic largest-remainder proportional attribution of a shared
    drop_total packets among this tick's per-flow arrivals."""
    n = len(arrivals)
    total = sum(arrivals)
    if drop_total <= 0 or total <= 0:
        return [0] * n
    drop_total = min(drop_total, total)
    shares = [drop_total * a for a in arrivals]
    floor_drops = [s // total for s in shares]
    remainders = [(shares[k] % total, k) for k in range(n)]
    remaining = drop_total - sum(floor_drops)
    order = sorted(remainders, key=lambda x: (-x[0], x[1]))
    drops = floor_drops[:]
    for k in range(remaining):
        drops[order[k][1]] += 1
    return [min(drops[k], arrivals[k]) for k in range(n)]


# ------------------------------------------------------------- simulation --
def simulate(ego_decider, inst):
    T, C, Qmax = inst['T'], inst['C'], inst['Qmax']
    comps = inst['comps']
    n_comp = len(comps)
    FAIR = max(1, C // (n_comp + 1))

    mem = [0, 0, 0, 0]
    cwnd = inst['init_cwnd']
    loss_flag = 0
    queue = 0
    comp_state = [{'win': c['p1']} if c['type'] == 'AIMD' else {} for c in comps]
    goodput = [0] * (n_comp + 1)
    sum_queue = 0

    for t in range(T):
        regs = [0] * REG_N
        regs[0], regs[1], regs[2], regs[3] = mem
        regs[4] = cwnd
        regs[5] = loss_flag
        regs[6] = queue                 # delay signal: backlog observed BEFORE this tick
        regs[7] = inst['base_rtt_ego']
        regs[8] = t

        new_cwnd, new_mem = ego_decider(regs)
        new_cwnd = max(0, min(CWND_CAP, new_cwnd))
        new_mem = [max(-CAP, min(CAP, x)) for x in new_mem]

        comp_sends = []
        for idx, c in enumerate(comps):
            if c['type'] == 'AIMD':
                comp_sends.append(comp_state[idx]['win'])
            elif c['type'] == 'CONST':
                comp_sends.append(c['p1'])
            elif c['type'] == 'ONOFF':
                period = c['p2'] + c['p3']
                phase = t % period if period > 0 else 0
                comp_sends.append(c['p1'] if phase < c['p2'] else 0)
            else:
                comp_sends.append(0)

        arrivals = [new_cwnd] + comp_sends
        total_arrival = sum(arrivals)
        served = min(C, queue + total_arrival)
        overflow = queue + total_arrival - served
        if overflow > Qmax:
            drop_total = overflow - Qmax
            queue_next = Qmax
        else:
            drop_total = 0
            queue_next = overflow
        drops = allocate_drops(arrivals, drop_total)

        for k in range(n_comp + 1):
            goodput[k] += arrivals[k] - drops[k]
        sum_queue += queue_next

        loss_flag_next = 1 if drops[0] > 0 else 0
        for idx, c in enumerate(comps):
            if c['type'] == 'AIMD':
                st = comp_state[idx]
                if drops[1 + idx] > 0:
                    st['win'] = max(1, st['win'] // 2)
                else:
                    st['win'] = st['win'] + c['p2']

        mem, cwnd, loss_flag, queue = new_mem, new_cwnd, loss_flag_next, queue_next

    # Per-tick RATE targets, scaled by T into ACCUMULATED targets so they are
    # comparable to accumulated goodput (a flow that averages its target rate
    # over the whole horizon should read as "fully served", not "served for
    # one tick out of T"). Scaling every target by the same T leaves the Jain
    # index (J) unchanged (scale-invariant), but is essential for U below.
    rate_targets = [FAIR] + [(c['demand'] if c['demand'] > 0 else FAIR) for c in comps]
    targets = [r * T for r in rate_targets]
    normalized = [goodput[k] / targets[k] for k in range(n_comp + 1)]
    nf = n_comp + 1
    if nf == 1:
        J = 1.0
    else:
        s1 = sum(normalized)
        s2 = sum(x * x for x in normalized)
        J = (s1 * s1) / (nf * s2) if s2 > 0 else 1.0

    avg_queue_ratio = sum_queue / (T * C)
    TARGET_TOTAL = sum(targets)
    G_total = sum(goodput)
    underutil = max(0.0, (TARGET_TOTAL - G_total)) / TARGET_TOTAL

    F = inst['ALPHA'] * (1.0 - J) + inst['BETA'] * avg_queue_ratio + inst['GAMMA'] * underutil
    return F


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

    def ego_decider(regs):
        work = regs[:]
        run_program(instrs, work)
        c = resolve(result_ops[0], work)
        m = [resolve(result_ops[k], work) for k in range(1, 5)]
        if c != c or c in (float('inf'), float('-inf')):
            c = 0
        return c, m

    def baseline_decider(regs):
        return inst['init_cwnd'], [0, 0, 0, 0]

    F = simulate(ego_decider, inst)
    B = simulate(baseline_decider, inst)

    if F != F or F in (float('inf'), float('-inf')) or F < 0:
        print("INVALID: non-finite or negative cost")
        print("Ratio: 0.0")
        return

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the panic-mode
context-aware synchronization problem.

Reads K token-programs from <in>. Reads the submitted 3-line sync-set table
from <out> (rejecting anything malformed -> Ratio 0.0). Runs a FIXED,
deterministic panic-mode recovery engine (identical rules for every
submission) over every program using the submitted table, and separately
using an empty-everywhere table (the internal baseline B). Score:

    TP = # reported errors whose position holds a true `?`
    FP = # reported errors whose position does not (a phantom)
    F  = TP - FP  (submitted table);  B = same quantity, empty table
    sc = min(1000, max(0, 100*F/B));  Ratio = sc/1000

Exact integer bookkeeping, O(total tokens), bit-for-bit deterministic.
"""
import sys

CODE_TO_TOK = {0: ';', 1: ',', 2: '(', 3: ')', 4: '{', 5: '}'}
TOK_TO_CODE = {v: k for k, v in CODE_TO_TOK.items()}
EMPTY_TABLE = [set(), set(), set()]
CTX_IDX = {'STMT': 0, 'PAREN': 1, 'FORHDR': 2}


def fail(reason):
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_instance(path):
    with open(path) as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError("empty instance")
    k = int(lines[0].split()[0])
    programs = []
    for i in range(k):
        toks = lines[1 + i].split() if 1 + i < len(lines) else []
        programs.append(toks)
    return programs


def parse_output(path):
    """Return a 3-set table [STMT, PAREN, FORHDR] of ints in [0,5], or None
    on ANY malformed / out-of-range / non-finite content (strict, adversarial-
    safe: int() parsing itself already rejects 'nan'/'inf'/garbage)."""
    try:
        with open(path) as f:
            txt = f.read()
    except Exception:
        return None
    lines = txt.splitlines()
    if len(lines) == 4 and lines[3].strip() == "":
        lines = lines[:3]
    if len(lines) != 3:
        return None
    table = []
    for ln in lines:
        toks = ln.split()
        if len(toks) > 6:
            return None
        s = set()
        for t in toks:
            try:
                v = int(t)
            except ValueError:
                return None
            if v < 0 or v > 5:
                return None
            s.add(v)
        table.append(s)
    return table


def run_engine(tokens, table):
    """Fixed panic-mode recovery engine. `table` = [set_stmt, set_paren,
    set_forhdr] of codes 0..5 (; , ( ) { }). Returns the list of token
    positions at which an error was reported (both true positives and
    phantoms -- caller classifies against the program's true `?` positions).
    Deterministic, single pass + bounded recursive unwind (<= nesting depth)."""
    n = len(tokens)
    stack = ['STMT']
    reports = []
    i = 0

    def sync_chars():
        codes = table[CTX_IDX[stack[-1]]]
        return {CODE_TO_TOK[c] for c in codes}

    def resolve():
        nonlocal i
        if i >= n:
            return
        t = tokens[i]
        top = stack[-1]
        base = len(stack) == 1
        if t == ')':
            if top in ('PAREN', 'FORHDR'):
                stack.pop(); i += 1
            elif not base:
                stack.pop(); resolve()
            else:
                i += 1
        elif t == '}':
            if top == 'STMT' and not base:
                stack.pop(); i += 1
            elif not base:
                stack.pop(); resolve()
            else:
                i += 1
        elif t == ';':
            if top in ('STMT', 'FORHDR'):
                i += 1
            elif not base:
                stack.pop(); resolve()
            else:
                i += 1
        elif t == ',':
            i += 1
        elif t in ('(', '{'):
            if not base:
                stack.pop(); resolve()
            else:
                i += 1
        else:
            i += 1  # defensive: never reached (sync targets are punctuation only)

    def enter_recovery():
        nonlocal i
        reports.append(i)
        S = sync_chars()
        j = i
        while j < n and tokens[j] not in S:
            j += 1
        i = j
        resolve()

    while i < n:
        t = tokens[i]
        if t == '(':
            stack.append('FORHDR' if (i > 0 and tokens[i - 1] == 'F') else 'PAREN')
            i += 1
        elif t == '{':
            stack.append('STMT')
            i += 1
        elif t == ')':
            if stack[-1] in ('PAREN', 'FORHDR'):
                stack.pop(); i += 1
            else:
                enter_recovery()
        elif t == '}':
            if stack[-1] == 'STMT' and len(stack) > 1:
                stack.pop(); i += 1
            else:
                enter_recovery()
        elif t == '?':
            enter_recovery()
        else:
            i += 1
    return reports


def score_table(programs, table):
    tp = 0
    fp = 0
    for toks in programs:
        true_pos = {idx for idx, t in enumerate(toks) if t == '?'}
        reports = run_engine(toks, table)
        for r in reports:
            if r in true_pos:
                tp += 1
            else:
                fp += 1
    return tp, fp


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        programs = parse_instance(in_path)
    except Exception as e:
        fail("bad instance: %r" % (e,))
    if not programs:
        fail("no programs in instance")

    table = parse_output(out_path)
    if table is None:
        fail("malformed output (need exactly 3 lines of distinct ints in [0,5])")

    tp, fp = score_table(programs, table)
    F = tp - fp

    btp, bfp = score_table(programs, EMPTY_TABLE)
    B = max(1, btp - bfp)

    sc = min(1000.0, max(0.0, 100.0 * F / B))
    print("K=%d TP=%d FP=%d F=%d B=%d" % (len(programs), tp, fp, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

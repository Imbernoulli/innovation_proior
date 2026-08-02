#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>

Scores a bracket of Claimer-win / Blocker-safe certificates over K nested
winning-line families L_1 subset ... subset L_K on an N-cell claiming game.

Both certificate kinds are mechanically self-verifying (no hidden ground
truth is needed): a valid weight vector proves Blocker-safety by a real
inequality (Erdos-Selfridge style potential bound), and a valid strategy
table proves a forced Claimer win by exhaustive adversarial replay. So any
level with a VALID certificate is, by construction of the check, a TRUE
statement about that level's game.
"""
import sys
from fractions import Fraction

MAX_TABLE_ROWS = 4000
NODE_CAP = 200000


def fail(reason):
    print(f"INVALID: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    pool = []
    for _ in range(P):
        sz = int(next(it))
        cells = tuple(int(next(it)) for _ in range(sz))
        pool.append(cells)
    c = [int(next(it)) for _ in range(K)]
    return N, K, P, pool, c


def levels_from_pool(pool, c):
    return [pool[:cj] for cj in c]


def check_b_cert(level_lines, weight_toks):
    if len(weight_toks) != len(level_lines):
        return False
    total = Fraction(0)
    for tok, line in zip(weight_toks, level_lines):
        if "/" not in tok:
            return False
        p_s, q_s = tok.split("/", 1)
        try:
            p = int(p_s)
            q = int(q_s)
        except ValueError:
            return False
        if p <= 0 or q <= 0:
            return False
        w = Fraction(p, q)
        s = len(line)
        # per-line floor: w * 2^s >= 1
        if w * (2 ** s) < 1:
            return False
        total += w
    # Claimer moves first in this game, so the Erdos-Selfridge potential
    # bound requires sum(w) < 1/2 (not < 1, which only holds when Blocker
    # moves first) for the mechanical potential-decrease strategy to be a
    # sound guarantee.
    return total < Fraction(1, 2)


def check_m_cert(N, level_lines, rows):
    line_sets = [frozenset(l) for l in level_lines]
    table = {}
    for row in rows:
        if len(row) < 2:
            return False
        try:
            d = int(row[0])
        except ValueError:
            return False
        if d < 0 or d % 2 != 0 or d > N:
            return False
        if len(row) != d + 2:
            return False
        try:
            hist = tuple(int(x) for x in row[1:1 + d])
            mv = int(row[1 + d])
        except ValueError:
            return False
        state = hist
        if state in table:
            return False  # ambiguous / duplicate state
        table[state] = mv

    nodes = [0]

    def claimer_owns_line(owned):
        for ls in line_sets:
            if ls <= owned:
                return True
        return False

    def dfs(hist, claimer, blocker):
        nodes[0] += 1
        if nodes[0] > NODE_CAP:
            return False
        depth = len(hist)
        if depth >= N:
            return False  # board full, claimer never completed a line
        if depth % 2 == 0:
            # claimer's turn -- must be in table
            state = hist
            if state not in table:
                return False
            mv = table[state]
            if mv < 1 or mv > N or mv in claimer or mv in blocker:
                return False
            new_claimer = claimer | {mv}
            new_hist = hist + (mv,)
            if claimer_owns_line(new_claimer):
                return True
            return dfs(new_hist, new_claimer, blocker)
        else:
            # blocker's turn -- checker enumerates EVERY legal reply
            used = claimer | blocker
            empties = [c for c in range(1, N + 1) if c not in used]
            if not empties:
                return False
            for mv in empties:
                ok = dfs(hist + (mv,), claimer, blocker | {mv})
                if not ok:
                    return False
            return True

    try:
        return dfs((), frozenset(), frozenset())
    except RecursionError:
        return False


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        N, K, P, pool, c = read_instance(in_path)
    except Exception as e:
        fail(f"bad instance file: {e}")

    levels = levels_from_pool(pool, c)

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception as e:
        fail(f"cannot read output: {e}")

    if not out_toks:
        fail("empty output")

    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(out_toks):
            raise IndexError("ran out of tokens")
        v = out_toks[pos]
        pos += 1
        return v

    try:
        Kp = int(nxt())
    except Exception:
        fail("first token must be integer K")

    if Kp != K:
        fail(f"K mismatch: expected {K} got {Kp}")

    b_set = set()
    m_set = set()

    try:
        for j in range(1, K + 1):
            jt = int(nxt())
            tag = nxt()
            if jt != j:
                fail(f"levels must be listed in order 1..K, got {jt} at slot {j}")
            if tag == "U":
                continue
            elif tag == "B":
                m_j = c[j - 1]
                toks = [nxt() for _ in range(m_j)]
                if check_b_cert(levels[j - 1], toks):
                    b_set.add(j)
            elif tag == "M":
                T = int(nxt())
                if T < 0 or T > MAX_TABLE_ROWS:
                    fail(f"table row count out of range at level {j}")
                rows = []
                for _ in range(T):
                    d = int(nxt())
                    if d < 0 or d > N:
                        fail("corrupt table row depth")
                    row = [str(d)] + [nxt() for _ in range(d + 1)]
                    rows.append(row)
                if check_m_cert(N, levels[j - 1], rows):
                    m_set.add(j)
            else:
                fail(f"unknown tag '{tag}' at level {j}")
    except (IndexError, ValueError) as e:
        fail(f"malformed / truncated output: {e}")

    if pos != len(out_toks):
        fail("trailing garbage tokens after the last level")

    S = len(b_set) + len(m_set)
    bonus = 0
    if b_set and m_set:
        l = max(b_set)
        h = min(m_set)
        if l < h:
            bonus = max(0, 4 - (h - l))
    F = S + bonus

    B0 = 1.0  # checker's own baseline: it can always certify level 1 alone
    sc = min(1000.0, 100.0 * F / max(1e-9, B0))
    print(f"levels_certified={S} bracket_bonus={bonus} F={F} B0={B0}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

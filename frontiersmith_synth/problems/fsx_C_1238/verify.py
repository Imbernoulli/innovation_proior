#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the flag-rollout problem.

Prints a line ending in 'Ratio: <float in [0,1]>' and exits 0.
"""
import sys
import math


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    pos = [0]

    def nxt():
        v = toks[pos[0]]
        pos[0] += 1
        return v

    N = int(nxt())
    values = [int(nxt()) for _ in range(N)]
    R = int(nxt())
    req_of = {i: [] for i in range(1, N + 1)}
    for _ in range(R):
        c = int(nxt())
        p = int(nxt())
        req_of[c].append(p)
    C = int(nxt())
    conf_of = {i: set() for i in range(1, N + 1)}
    for _ in range(C):
        a = int(nxt())
        b = int(nxt())
        conf_of[a].add(b)
        conf_of[b].add(a)
    return N, values, req_of, conf_of


def parse_output(path, N):
    try:
        with open(path) as f:
            raw_lines = [ln.strip() for ln in f]
    except Exception:
        return None
    lines = [ln for ln in raw_lines if ln != '']
    if len(lines) != N:
        return None
    actions = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 1 and parts[0].upper() == 'P':
            actions.append(('P', None))
        elif len(parts) == 2 and parts[0].upper() in ('E', 'R'):
            try:
                x = int(parts[1])
            except Exception:
                return None
            if not math.isfinite(x):
                return None
            actions.append((parts[0].upper(), x))
        else:
            return None
    return actions


def simulate(actions, N, values, req_of, conf_of):
    """Returns (total_value, error_reason_or_None)."""
    active = set()
    ever_enabled = set()
    ever_disabled = set()
    total = 0
    for (typ, x) in actions:
        if typ == 'E':
            if x < 1 or x > N:
                return None, "flag id out of range"
            if x in ever_enabled or x in ever_disabled:
                return None, "flag re-touched after being enabled/rolled back"
            for p in req_of[x]:
                if p not in active:
                    return None, "requires-flag not active at enable time"
            for y in conf_of[x]:
                if y in active:
                    return None, "conflicting flag active at enable time"
            active.add(x)
            ever_enabled.add(x)
        elif typ == 'R':
            if x < 1 or x > N:
                return None, "flag id out of range"
            if x not in active:
                return None, "rollback of a flag that is not active"
            active.discard(x)
            ever_disabled.add(x)
        elif typ == 'P':
            pass
        else:
            return None, "unrecognized action token"
        total += sum(values[i - 1] for i in active)
    return total, None


def baseline_value(N, values, req_of, conf_of):
    """Weak deterministic construction: enable only flags with NO requires and
    NO conflicts at all, in index order; everything else is a pass."""
    isolated = set(
        i for i in range(1, N + 1)
        if len(req_of[i]) == 0 and len(conf_of[i]) == 0
    )
    actions = []
    for i in range(1, N + 1):
        if i in isolated:
            actions.append(('E', i))
        else:
            actions.append(('P', None))
    total, err = simulate(actions, N, values, req_of, conf_of)
    assert err is None, "internal baseline construction must be feasible"
    return max(total, 1e-9)


def main():
    if len(sys.argv) < 3:
        print("Bad invocation. Ratio: 0.0")
        return
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, values, req_of, conf_of = read_input(in_path)

    actions = parse_output(out_path, N)
    if actions is None:
        print("Malformed output (wrong line count / bad token / non-finite). Ratio: 0.0")
        return

    F, err = simulate(actions, N, values, req_of, conf_of)
    if err is not None:
        print(f"Infeasible: {err}. Ratio: 0.0")
        return

    B = baseline_value(N, values, req_of, conf_of)
    # BASELINE_MULT calibrated (empirically, against the reference solution
    # ladder) so that matching the weak baseline scores ~0.07-0.1, and the
    # strong reference solution never saturates (headroom stays open above
    # it even on the largest/most tightly-coupled instance).
    BASELINE_MULT = 14.0
    ratio = min(1.0, F / (BASELINE_MULT * max(1e-9, B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == '__main__':
    main()

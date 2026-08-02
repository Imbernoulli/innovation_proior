#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the supplier-ESG
audit-route problem. Prints 'Ratio: <float in [0,1]>' on its own last line.

Feasibility (validated strictly; any violation -> Ratio: 0.0):
  - every action line is exactly "OP ID" with OP in {M, A} (case-insensitive)
    and ID a finite integer token in [1, N];
  - "M id": id's tier is 1 or 2, id is currently VISIBLE, and id is not
    already mapped;
  - "A id": id is currently VISIBLE and not already audited;
  - a node is visible iff it is tier-1, or its parent has been mapped by an
    earlier action in the sequence;
  - running total cost (map/audit costs, integers) never exceeds the budget.

Score: total risk-exposure reduced (audited nodes get full AUDITMIT
mitigation, mapped-only tier1/2 nodes get partial MAPMIT mitigation, tier3
nodes get nothing until audited), each node's raw risk discounted upstream
by PROP^(tier-1) before being counted -- matching the statement. Normalized
against an internal baseline B (audit only the single highest-raw-risk
tier-1 supplier, the "obvious first move").
"""
import math
import sys


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def parse_input(path):
    toks = read_tokens(path)
    idx = 0
    n = int(toks[idx]); idx += 1
    t1n = int(toks[idx]); idx += 1
    budget = int(toks[idx]); idx += 1
    prop = float(toks[idx]); idx += 1
    mapmit = float(toks[idx]); idx += 1
    auditmit = float(toks[idx]); idx += 1
    tier = {}
    parent = {}
    risk = {}
    mapcost = {}
    auditcost = {}
    children = {}
    for _ in range(n):
        nid = int(toks[idx]); idx += 1
        t = int(toks[idx]); idx += 1
        p = int(toks[idx]); idx += 1
        r = int(toks[idx]); idx += 1
        mc = int(toks[idx]); idx += 1
        ac = int(toks[idx]); idx += 1
        tier[nid] = t
        parent[nid] = p
        risk[nid] = r
        mapcost[nid] = mc
        auditcost[nid] = ac
        children.setdefault(nid, [])
        if p:
            children.setdefault(p, []).append(nid)
    return dict(n=n, t1n=t1n, budget=budget, prop=prop, mapmit=mapmit,
                auditmit=auditmit, tier=tier, parent=parent, risk=risk,
                mapcost=mapcost, auditcost=auditcost, children=children)


def factor(inst, t):
    return inst["prop"] ** (t - 1)


def val_audit(inst, v):
    return inst["auditmit"] * inst["risk"][v] * factor(inst, inst["tier"][v])


def val_map(inst, v):
    return inst["mapmit"] * inst["risk"][v] * factor(inst, inst["tier"][v])


def exposure_reduced(inst, mapped, audited):
    tot = 0.0
    for v in inst["tier"]:
        if v in audited:
            tot += val_audit(inst, v)
        elif v in mapped and inst["tier"][v] in (1, 2):
            tot += val_map(inst, v)
    return tot


def baseline_B(inst):
    t1ids = [v for v in inst["tier"] if inst["tier"][v] == 1]
    best = max(t1ids, key=lambda v: (inst["risk"][v], -v))
    if inst["auditcost"][best] <= inst["budget"]:
        return exposure_reduced(inst, set(), {best})
    return 1e-6


def parse_output_actions(path, n):
    try:
        with open(path) as f:
            lines = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    except Exception as e:
        fail(f"cannot read output: {e}")
    if len(lines) > 3 * n + 5:
        fail("too many action lines")
    actions = []
    for toks in lines:
        if len(toks) != 2:
            fail(f"malformed action line: {toks}")
        op = toks[0].upper()
        if op not in ("M", "A"):
            fail(f"unknown op: {toks[0]}")
        try:
            nid = int(toks[1])
        except ValueError:
            fail(f"non-integer id: {toks[1]}")
        if not (1 <= nid <= n):
            fail(f"id out of range: {nid}")
        actions.append((op, nid))
    return actions


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    n, budget = inst["n"], inst["budget"]
    tier, parent, children = inst["tier"], inst["parent"], inst["children"]
    mapcost, auditcost = inst["mapcost"], inst["auditcost"]

    actions = parse_output_actions(outp, n)

    visible = set(v for v in tier if tier[v] == 1)
    mapped, audited = set(), set()
    used = 0
    for op, v in actions:
        if op == "M":
            if tier[v] not in (1, 2):
                fail(f"cannot map a tier-{tier[v]} (leaf) node {v}")
            if v not in visible:
                fail(f"node {v} is not yet visible (parent unmapped)")
            if v in mapped:
                fail(f"node {v} mapped twice")
            c = mapcost[v]
            used += c
            if used > budget:
                fail("budget exceeded on a map action")
            mapped.add(v)
            for ch in children.get(v, []):
                visible.add(ch)
        else:
            if v not in visible:
                fail(f"node {v} is not yet visible (parent unmapped)")
            if v in audited:
                fail(f"node {v} audited twice")
            c = auditcost[v]
            used += c
            if used > budget:
                fail("budget exceeded on an audit action")
            audited.add(v)

    F = exposure_reduced(inst, mapped, audited)
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative objective")

    B = baseline_B(inst)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"exposure_reduced={F:.4f} baseline={B:.4f} used_budget={used}/{budget}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()

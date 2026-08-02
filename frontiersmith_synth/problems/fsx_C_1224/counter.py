#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for the wire-tag
reservation problem (format D / eval_form=flops: the "op count" here is the
encoded byte count of the wire schema across its whole multi-version
lifetime; the "exact equivalence" gate is the forward/backward-compat
validity check on the tag assignment).

Feasibility (ANY violation -> Ratio: 0.0):
  - the participant output must be EXACTLY 2*M whitespace-separated integer
    tokens, read as M (field_id, tag) pairs
  - the field_ids must be a permutation of 0..M-1 (every field gets exactly
    one tag, no field skipped or duplicated)
  - every tag must be a finite integer in [0, TAG_MAX]
  - all M tags must be pairwise DISTINCT -- two different fields sharing one
    tag is exactly the incompatible-version-pair break: a parser cannot tell
    which field a byte with that tag belongs to across versions.

Objective (maximize the ratio):
  F = total encoded bytes = sum over fields of freq * (versions the field is
      active) * tag_cost(tag), where tag_cost is the fixed step schedule
      (T1CAP tags @ 1 byte, next T2CAP tags @ T2COST bytes, rest @ T3COST
      bytes -- an expensive "overflow key" fallback).
  B = the SAME cost formula applied to the checker's own reference
      assignment: sort all M fields by ASCENDING freq and hand out tags
      0..M-1 in that order (the least-used fields get the cheap tags).
  sc = min(1.0, B / (10 * F))   (fewer bytes -> higher score; matches trivial
       reproducing B -> ~0.1, a 10x-leaner schema saturates at 1.0)
"""
import sys, math

TAG_MAX = 999_999


def tag_cost(tag, t1cap, t2cap, t2cost, t3cost):
    if tag < t1cap:
        return 1
    if tag < t1cap + t2cap:
        return t2cost
    return t3cost


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    V, M, t1cap, t2cap, t2cost, t3cost = (int(next(it)) for _ in range(6))
    fields = []  # (group, v0, freq) indexed by field_id
    for _ in range(M):
        fid = int(next(it)); g = int(next(it)); v0 = int(next(it)); freq = int(next(it))
        fields.append((v0, freq))
    return V, M, t1cap, t2cap, t2cost, t3cost, fields


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    V, M, t1cap, t2cap, t2cost, t3cost, fields = read_instance(in_path)

    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("cannot read output")

    if len(out_toks) != 2 * M:
        fail(f"expected {2*M} tokens (M field_id,tag pairs), got {len(out_toks)}")

    tag_of = {}
    for k in range(M):
        ftok, ttok = out_toks[2 * k], out_toks[2 * k + 1]
        try:
            fid = int(ftok)
            tag = int(ttok)
        except ValueError:
            fail(f"non-integer token pair ({ftok!r}, {ttok!r})")
        if not (math.isfinite(fid) and math.isfinite(tag)):
            fail("non-finite token")
        if fid < 0 or fid >= M:
            fail(f"field_id {fid} out of range [0,{M})")
        if fid in tag_of:
            fail(f"field_id {fid} repeated")
        if tag < 0 or tag > TAG_MAX:
            fail(f"tag {tag} out of range [0,{TAG_MAX}]")
        tag_of[fid] = tag

    if len(tag_of) != M:
        fail("not all field_ids covered")

    tags_seen = set()
    for fid in range(M):
        t = tag_of[fid]
        if t in tags_seen:
            fail(f"tag {t} reused by two different fields -- incompatible version pair")
        tags_seen.add(t)

    F = 0
    for fid in range(M):
        v0, freq = fields[fid]
        active_versions = V - v0 + 1
        F += freq * active_versions * tag_cost(tag_of[fid], t1cap, t2cap, t2cost, t3cost)

    # reference: sort ALL fields by ascending freq, hand out tags 0..M-1 in
    # that order (rarest fields get the cheap tags -- a fixed, trivially
    # constructible, deliberately unhelpful anchor for normalization).
    ref_order = sorted(range(M), key=lambda i: fields[i][1])
    ref_tag = {fid: rank for rank, fid in enumerate(ref_order)}
    B = 0
    for fid in range(M):
        v0, freq = fields[fid]
        active_versions = V - v0 + 1
        B += freq * active_versions * tag_cost(ref_tag[fid], t1cap, t2cap, t2cost, t3cost)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print(f"F={F} B={B} Ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()

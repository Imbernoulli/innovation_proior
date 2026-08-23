#!/usr/bin/env python3
"""Offline replay of the banked failed edit attempts against the new contract.

Two separate questions, reported separately because they have very different
evidential strength:

  (1) MECHANICAL -- given the exact file bytes at that moment and the exact
      old_str the model sent, does the new matcher now resolve it to a unique,
      in-range span? This is a sound, model-free claim: same input, same code
      path, deterministic answer.

  (2) ROUTING -- for the failures that the matcher still cannot rescue, does the
      new contract contain an operation that expresses the same intent and
      cannot fail? Concretely: an anchor covering most of the editable region is
      the model transcribing the region in order to replace it, which op='rewrite'
      does with no anchor at all. This is NOT a claim that the model will choose
      rewrite -- only that a deterministic path to success now exists where none
      did. Reported as an upper bound, separately from (1).
"""
import collections
import json
import pickle
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "repo"))
from mlsbench.agent.tools import resolve_old_str  # noqa: E402

F = pickle.load(open(HERE / "failures2.pkl", "rb"))
print(f"{len(F)} banked 'old_str not found' calls with reconstructible file state\n")

RANGE_RE = re.compile(r"\| editable: ([^|]+)\|")


def editable_ranges(result):
    m = RANGE_RE.search(result or "")
    if not m:
        return None
    out = []
    for part in m.group(1).split(","):
        part = part.strip().replace("–", "-").replace("—", "-")
        mm = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if mm:
            out.append((int(mm.group(1)), int(mm.group(2))))
    return out or None


mech = collections.Counter()
lvl = collections.Counter()
route = collections.Counter()
cov_hist = collections.Counter()
by_task = collections.defaultdict(collections.Counter)

for i, r in enumerate(F):
    if i % 1000 == 0:
        print(f"  {i}/{len(F)}", file=sys.stderr, flush=True)
    text, old = r["text"], r["old"]
    er = editable_ranges(r.get("result"))
    span = sum(e - s + 1 for s, e in er) if er else 0
    n_anchor = len(old.split("\n"))
    frac = n_anchor / span if span else 0.0

    # ---- (1) mechanical -------------------------------------------------
    m = resolve_old_str(text, old)
    if m["ok"]:
        if er is None:
            tag = "RESCUED_range_unknown"
        elif any(s <= m["line_start"] and m["line_end"] <= e for s, e in er):
            tag = "RESCUED_in_range"
        else:
            tag = "matched_but_out_of_editable_range"
        lvl[m["level"]] += 1
    else:
        tag = "still_" + m["reason"]
    mech[tag] += 1
    by_task[r["task"]][tag] += 1

    # ---- (2) routing ----------------------------------------------------
    rescued = tag.startswith("RESCUED")
    if rescued:
        route["already_fixed_by_matcher"] += 1
    elif er is not None and len(er) == 1 and frac >= 0.6:
        route["rewrite_eligible_region_scale"] += 1
    elif tag == "matched_but_out_of_editable_range" and er is not None and len(er) == 1:
        route["rewrite_eligible_out_of_range"] += 1
    elif any(old in p for p in r.get("priors") or []):
        route["stale_prior_version_echo_now_shows_current"] += 1
    else:
        route["needs_the_model_to_read_view_or_echo"] += 1

    b = ("<10%" if frac < .1 else "10-30%" if frac < .3 else "30-60%" if frac < .6
         else "60-95%" if frac < .95 else ">=95%")
    cov_hist[b] += 1

tot = len(F)
print(f"\n=== (1) MECHANICAL: new matcher replayed on {tot} banked failures ===")
for k, v in mech.most_common():
    print(f"  {v:6d}  ({100 * v / tot:5.1f}%)  {k}")
print("\n  match level used for the rescues:")
for k, v in lvl.most_common():
    print(f"    {v:6d}  {k}")
resc = mech["RESCUED_in_range"] + mech["RESCUED_range_unknown"]
print(f"\n  accepted by the matcher alone: {resc}/{tot} = {100 * resc / tot:.1f}%")

print(f"\n=== (2) ROUTING: is there now a deterministic path for the rest? ===")
for k, v in route.most_common():
    print(f"  {v:6d}  ({100 * v / tot:5.1f}%)  {k}")

print("\n=== anchor size as a fraction of the editable region (these failures) ===")
for k in ["<10%", "10-30%", "30-60%", "60-95%", ">=95%"]:
    v = cov_hist.get(k, 0)
    print(f"  {k:>7}  {v:6d}  ({100 * v / tot:5.1f}%)")

# ---- project onto the whole str_replace population -----------------------
stats = json.load(open(HERE / "collect_stats.json"))
calls, ok, nf = stats["str_replace_calls"], stats["str_replace_ok"], stats["notfound"]
scale = nf / tot
print(f"\n=== projected over all {calls} banked str_replace calls ===")
print(f"  ({nf} were 'not found'; {tot} reconstructible, scale x{scale:.2f})")
print(f"  before               : {ok:5d}/{calls} accepted = {100 * ok / calls:.1f}%  "
      f"(rejection {100 * (calls - ok) / calls:.1f}%)")
a1 = ok + resc * scale
print(f"  + matcher (mechanical): {a1:5.0f}/{calls} accepted = {100 * a1 / calls:.1f}%  "
      f"(rejection {100 * (calls - a1) / calls:.1f}%)")
rw = (route["rewrite_eligible_region_scale"] + route["rewrite_eligible_out_of_range"]) * scale
a2 = a1 + rw
print(f"  + rewrite routing    : {a2:5.0f}/{calls} accepted = {100 * a2 / calls:.1f}%  "
      f"(rejection {100 * (calls - a2) / calls:.1f}%)   [upper bound]")

json.dump({"mech": dict(mech), "lvl": dict(lvl), "route": dict(route),
           "cov": dict(cov_hist), "n": tot, "scale": scale,
           "calls": calls, "ok": ok},
          open(HERE / "replay_contract.json", "w"), indent=1)
print("\nwrote replay_contract.json")

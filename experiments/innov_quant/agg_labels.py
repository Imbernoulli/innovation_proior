"""Aggregate blind labels: join labels/*.jsonl with key.json; grep-check quotes; tables by bench x pair-kind x side."""
import json, glob, os, collections
B = "blind"; key = json.load(open(f"{B}/key.json"))
labs = {}
bad_quote = 0; n_lab = 0
for f in sorted(glob.glob(f"{B}/labels/*.jsonl")):
    for l in open(f):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception as e: print("BAD JSON", f, l[:80]); continue
        n_lab += 1
        fol = r["folder"].strip("/")
        if fol not in key: print("unknown folder", fol); continue
        ok = True
        for letter, q in (r.get("quotes") or {}).items():
            p = f"{B}/{fol}/{letter}.txt"
            if not os.path.exists(p) or q not in open(p).read(): ok = False
        if not ok: bad_quote += 1
        r["_quote_ok"] = ok; labs[fol] = r
print(f"labels {n_lab}, folders labeled {len(labs)}/{len(key)}, quote failures {bad_quote}")
# per folder: identify arm letter / ctrl letter
def letters(fol):
    k = key[fol]; m = {c: v["arm"] for c, v in k.items() if isinstance(v, dict) and "arm" in v}
    bench, pair, prob = fol.split("/", 2); a, c = pair.split("__vs__")
    la = next((c_ for c_, arm in m.items() if arm == a), None); lc = next((c_ for c_, arm in m.items() if arm == c), None)
    return bench, a, c, la, lc, k["winner_side"], k.get("kind", "main")
agg = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for fol, r in labs.items():
    if not r["_quote_ok"]: continue
    bench, a, c, la, lc, side, kind = letters(fol)
    if not la or not lc: continue
    winner, loser = (la, lc) if side == "arm" else (lc, la)
    from dump2 import ARMS
    stage_a = ARMS.get(a, ("?", "?"))[1]
    g = (bench, kind, stage_a, side)  # e.g. (frontiercs, main, rl_sft, arm)
    C = agg[g]; C["n"] += 1
    C["same_core_idea"] += bool(r.get("same_core_idea"))
    nov = r.get("novel") or {}; cd = r.get("cross_domain") or {}
    C[f"winner_novel={nov.get(winner)}"] += 1; C[f"loser_novel={nov.get(loser)}"] += 1
    C[f"why={r.get('why_higher_scores')}"] += 1
    C["winner_cross_domain"] += bool(cd.get(winner)); C["loser_cross_domain"] += bool(cd.get(loser))
    C["winner_different_loser_not"] += (nov.get(winner) == "different" and nov.get(loser) in ("textbook", "variant"))
    if nov.get(winner) == "different" and r.get("why_higher_scores") == "different_method":
        examples[(bench, kind, stage_a, side)].append((fol, winner, (r.get("approach") or {}).get(winner, ""), (r.get("quotes") or {}).get(winner, "")))
L = ["| bench | kind | arm stage | winner side | n | same core idea | winner: textbook/variant/different/none | loser: textbook/variant/different/none | why: diff_method / fewer_bugs / tuning / other_crashed / unclear | winner different & loser not | cross-domain winner/loser |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for g in sorted(agg):
    C = agg[g]; n = C["n"]
    nv = lambda w: "/".join(str(C[f"{w}_novel={x}"]) for x in ("textbook", "variant", "different", "none"))
    why = "/".join(str(C[f"why={x}"]) for x in ("different_method", "same_method_fewer_bugs", "same_method_better_tuning", "other_side_crashed_or_empty", "unclear"))
    L.append(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {n} | {C['same_core_idea']} | {nv('winner')} | {nv('loser')} | {why} | {C['winner_different_loser_not']} | {C['winner_cross_domain']}/{C['loser_cross_domain']} |")
L.append(""); L.append("## examples: winner labeled different + why=different_method"); 
for g, ex in sorted(examples.items()):
    L.append(f"### {g}")
    for fol, w, appr, q in ex: L.append(f"- `{fol}` [{w}] {appr} — quote: `{q[:160]}`")
open(f"{B}/label_tables.md", "w").write("\n".join(L)); print("\n".join(L))

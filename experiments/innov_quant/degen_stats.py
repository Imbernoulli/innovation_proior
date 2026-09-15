"""Per arm x bench: share of draws that (a) never reach </think> (thinking truncated), (b) reach </think> but hit the 32768 cap inside the
answer (answer truncated; fence unclosed), (c) have degenerate repetition in the code (unique-line fraction < 0.5 with >= 20 lines).
Also the score of each class. Common problems only. Writes degen_tables.md."""
import json, collections
from dump2 import ARMS
rows = collections.defaultdict(lambda: collections.Counter()); sc = collections.defaultdict(lambda: collections.defaultdict(float))
for l in open("samples.jsonl"):
    s = json.loads(l)
    if not s["in_common"]: continue
    k = (s["bench"], s["fam"], s["arm"]); c = rows[k]; c["n"] += 1
    lines = [x for x in s["code"].splitlines() if x.strip()]
    rep = s["has_code"] and len(lines) >= 20 and len(set(lines)) / len(lines) < 0.5
    cls = "think_trunc" if not s["complete"] else ("answer_trunc" if s["trunc"] else ("repetitive" if rep else "clean"))
    if s["complete"] and s["trunc"] and rep: c["answer_trunc_and_rep"] += 1
    c[cls] += 1; sc[k][cls] += s["score"]
L = []
for bench in ("frontiercs", "frontiercs_research", "alebench"):
    L.append(f"## {bench}\n"); L.append("| arm | stage | n draws | thinking truncated | answer truncated (cap hit inside code) | of which repetitive | repetitive but finished | clean | mean score: clean / answer-trunc / think-trunc |"); L.append("|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        k = (bench, ARMS[arm][0], arm); c = rows.get(k)
        if not c: continue
        n = c["n"]; f = lambda x: f"{100*c[x]/n:.0f}%"
        m = lambda x: f"{sc[k][x]/c[x]:.1f}" if c[x] else "–"
        L.append(f"| {arm} | {ARMS[arm][1]} | {n} | {f('think_trunc')} | {f('answer_trunc')} | {c['answer_trunc_and_rep']} | {f('repetitive')} | {f('clean')} | {m('clean')} / {m('answer_trunc')} / {m('think_trunc')} |")
    L.append("")
open("degen_tables.md", "w").write("\n".join(L)); print("\n".join(L))

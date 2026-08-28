"""Assemble the distilled corpus that the 4B arm actually trains on.

The controlled comparison against wd01 only holds if the two corpora differ in
ONE way: the wording of each trained turn's <think>. So this must emit exactly
the same rows, in the same order, with the same schema — sampling failures fall
back to the hand-written think rather than dropping the row, because a shorter
corpus would confound the arm with a data-size change.

Prints the fallback rate, which is the number to quote alongside any result: an
arm that fell back on half its turns is half an ablation.
"""
import argparse, json, os, re, sys

THINK = re.compile(r"<think>(.*?)</think>", re.S)
DROP_KEYS = ("_id", "_n_regen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--distill", default=".cache/tinker/innovation_distilled.jsonl")
    ap.add_argument("--out", default="/srv/home/bohanlyu/LF-innov/data/innovation_v2_distill.jsonl")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="reject a regenerated think shorter than this and fall back")
    a = ap.parse_args()

    dist = {}
    if os.path.exists(a.distill):
        for line in open(a.distill):
            if not line.strip():
                continue
            r = json.loads(line)
            dist[r["_id"]] = r
    print(f"[finalize] {len(dist)} regenerated rows on disk")

    n_rows = n_turns = n_regen = n_short = 0
    with open(a.out, "w") as f:
        for i, line in enumerate(open(a.orig)):
            if not line.strip():
                continue
            rid = f"{i:05d}"
            orig = json.loads(line)
            d = dist.get(rid)
            out = orig
            if d is not None:
                merged = json.loads(json.dumps(orig))
                ok = 0
                for j, m in enumerate(merged["conversations"]):
                    if m["from"] not in ("gpt", "function_call") or not m.get("loss"):
                        continue
                    om = THINK.search(m["value"])
                    if not om or j >= len(d["conversations"]):
                        continue
                    dm = THINK.search(d["conversations"][j]["value"])
                    if not dm:
                        continue
                    if len(dm.group(1).strip()) < a.min_chars:
                        n_short += 1
                        continue                       # keep the hand-written one
                    m["value"] = d["conversations"][j]["value"]
                    ok += 1
                out = merged
                n_regen += ok
            for k in DROP_KEYS:
                out.pop(k, None)
            # schema must match innov_v2 exactly
            assert set(out) == {"conversations", "system", "tools"}, sorted(out)
            n_turns += sum(1 for m in out["conversations"]
                           if m["from"] in ("gpt", "function_call") and m.get("loss")
                           and THINK.search(m["value"]))
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_rows += 1

    print(f"[finalize] {a.out}")
    print(f"[finalize] rows {n_rows}  (must equal the source row count)")
    print(f"[finalize] trained turns with a <think>: {n_turns}")
    print(f"[finalize] regenerated: {n_regen} ({n_regen/max(1,n_turns)*100:.1f}%)  "
          f"fell back: {n_turns-n_regen} ({(n_turns-n_regen)/max(1,n_turns)*100:.1f}%)  "
          f"[{n_short} of those were too-short generations]")


if __name__ == "__main__":
    main()

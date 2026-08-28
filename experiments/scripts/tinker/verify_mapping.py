"""Self-check for the invariant sample_inkling.py silently depends on.

The sampler regenerates a turn by its index in the rendered `msgs` list, then
writes the result back into `conversations` at index `i - off`, where `off`
accounts for the leading system message that `msgs` has and `conversations` does
not. If that mapping is ever off by one — a row with an empty system string, a
turn type to_oai() collapses or expands — the regenerated reasoning lands on the
WRONG turn and the corpus is silently corrupted in a way no downstream metric
would flag.

Run this after any change to to_oai() or the sampler's write-back.

    python3 experiments/scripts/tinker/verify_mapping.py     # exits non-zero on any error
"""
import argparse, json, sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from build_data import to_oai, THINK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    a = ap.parse_args()

    bad = checked = agentic = rows = 0
    for i, line in enumerate(open(a.src)):
        if not line.strip():
            continue
        rows += 1
        r = json.loads(line)
        msgs, flags = to_oai(r)
        if msgs is None:
            continue
        off = 1 if (r.get("system") and msgs and msgs[0]["role"] == "system") else 0
        if len(msgs) - off != len(r["conversations"]):
            print(f"row {i}: LENGTH MISMATCH msgs={len(msgs)} off={off} "
                  f"conv={len(r['conversations'])}")
            bad += 1
            continue
        for mi, m in enumerate(msgs):
            if m["role"] != "assistant" or not flags[mi]:
                continue
            conv = r["conversations"][mi - off]
            checked += 1
            if conv["from"] not in ("gpt", "function_call") or not conv.get("loss"):
                print(f"row {i} msg {mi} -> conv {mi-off}: ROLE MISMATCH "
                      f"{conv['from']} loss={conv.get('loss')}")
                bad += 1
                continue
            mt = THINK.search(conv["value"])
            if mt and m.get("reasoning_content") and \
                    mt.group(1).strip() != m["reasoning_content"].strip():
                print(f"row {i} msg {mi}: THINK CONTENT MISMATCH")
                bad += 1
        if any(t["from"] == "function_call" for t in r["conversations"]):
            agentic += 1

    print(f"rows {rows} ({agentic} agentic), trained assistant turns checked {checked}, "
          f"mapping errors {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

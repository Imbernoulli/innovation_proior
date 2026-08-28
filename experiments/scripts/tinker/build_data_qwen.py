"""Render our corpora into Qwen3.8 token Datums for Tinker LoRA training.

Differs from build_data.py (Inkling) in one load-bearing way: Qwen's chat template
ALWAYS emits a `<think>...</think>` pair for an assistant turn, filling it from the
`reasoning_content` field. Our rows carry the think as literal text *inside* content,
so passing them through unchanged yields

    <think>\n\n</think>\n\n<think>REAL THINK</think>ANSWER

— an empty think followed by a second one. That is the fold_think landmine. So the
think must be split out of content and handed to `reasoning_content`:

    <think>\nREAL THINK\n</think>\n\nANSWER

Everything else (per-turn loss mask, incremental rendering, agentic tool turns)
follows build_data.py.
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import THINK, split_think, parse_tool_call


def to_qwen(row):
    """LLaMA-Factory row -> (qwen messages, per-message train flag).

    Rows that carry explicit per-turn `loss` flags (the innovation corpus) are
    folded by those flags. Rows with no `loss` key anywhere (the maintain corpus)
    get LLaMA-Factory's default: every assistant turn is trained. Without this the
    maintain half renders with zero trained tokens and is silently dropped.
    """
    has_flags = any("loss" in t for t in row["conversations"])
    msgs, flags = [], []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]}); flags.append(False)
    pending = None
    for t in row["conversations"]:
        fr, v = t["from"], t["value"]
        loss = bool(t.get("loss")) if has_flags else True
        if fr == "human":
            msgs.append({"role": "user", "content": v}); flags.append(False)
        elif fr == "gpt":
            think, ans = split_think(v)
            m = {"role": "assistant", "content": ans}
            if think:
                m["reasoning_content"] = think          # <- the whole point
            msgs.append(m); flags.append(loss)
        elif fr == "function_call":
            think, visible, calls = parse_tool_call(v)
            if calls is None:
                return None, None
            m = {"role": "assistant", "content": visible or "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                                for c in calls]}
            if think:
                m["reasoning_content"] = think
            msgs.append(m); flags.append(loss)
            pending = calls[0]["name"]
        elif fr == "observation":
            msgs.append({"role": "tool", "name": pending or "tool", "content": v})
            flags.append(False)
        else:
            return None, None
    return msgs, flags


def render_spans(tok, msgs, flags, tools):
    kw = dict(tokenize=False)
    if tools:
        kw["tools"] = tools
    # Qwen's template raises "No user query found" on a system-only prefix, so the
    # incremental render starts at the first prefix that contains a user turn; that
    # whole prefix is context (weight 0) by construction.
    start = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if start is None:
        return None, None
    ids, w, prev_txt, prev_ids = [], [], "", []
    for i in range(start + 1, len(msgs) + 1):
        txt = tok.apply_chat_template(msgs[:i], **kw)
        if not txt.startswith(prev_txt):
            return None, None
        cur = tok.encode(txt)
        delta = cur[len(prev_ids):]
        if len(delta) <= 0:
            return None, None
        ids.extend(delta); w.extend([1.0 if flags[i - 1] else 0.0] * len(delta))
        prev_txt, prev_ids = txt, cur
    return ids, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="repeatable; each is a LLaMA-Factory jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--holdout", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260828)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    kept, drop = [], {"unparseable": 0, "too_long": 0, "render": 0, "no_train": 0}
    for src in a.src:
        tag = os.path.basename(src).split(".")[0]
        n0 = len(kept)
        for i, line in enumerate(open(src)):
            if not line.strip():
                continue
            r = json.loads(line)
            msgs, flags = to_qwen(r)
            if msgs is None:
                drop["unparseable"] += 1; continue
            tools = None
            if (r.get("tools") or "").strip():
                try:
                    tools = json.loads(r["tools"])
                except Exception:
                    drop["unparseable"] += 1; continue
            try:
                ids, w = render_spans(tok, msgs, flags, tools)
            except Exception:
                drop["render"] += 1; continue
            if ids is None:
                drop["render"] += 1; continue
            if len(ids) > a.max_len:
                drop["too_long"] += 1; continue
            if sum(w) == 0:
                drop["no_train"] += 1; continue
            kept.append({"id": f"{tag}:{i:05d}", "src": tag, "ids": ids, "w": w,
                         "n": len(ids), "n_train": int(sum(w))})
            if (i + 1) % 500 == 0:
                print(f"[build] {tag} {i+1} kept={len(kept)} {drop}", file=sys.stderr, flush=True)
        print(f"[build] {tag}: +{len(kept)-n0} rows", file=sys.stderr, flush=True)

    print(f"[build] kept {len(kept)} dropped {drop}", file=sys.stderr)
    import random
    rng = random.Random(a.seed); rng.shuffle(kept)
    hold, train = kept[: a.holdout], kept[a.holdout:]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for name, part in (("train", train), ("holdout", hold)):
        p = a.out.replace(".jsonl", f".{name}.jsonl")
        with open(p, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        import collections
        by = collections.Counter(r["src"] for r in part)
        print(f"[build] {p}: {len(part)} rows {dict(by)}, "
              f"{sum(r['n'] for r in part)/1e6:.2f}M tok, "
              f"{sum(r['n_train'] for r in part)/1e6:.2f}M trained", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Render the innovation SFT corpus into Inkling token Datums for Tinker LoRA training.

Source is the EXACT corpus the 4B arm was trained on:
    experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl   (2901 rows)
LLaMA-Factory shape: {system, tools, conversations:[{from, value, loss}]}
  from: human | gpt | function_call | observation
  loss: per-turn train/mask flag (the "folding") — masked turns are context only.

Inkling wire format (from its chat template):
    <|message_system|><|content_text|>SYS<|end_message|>
    <|message_system|>tool_declare<|content_xml|>[specs]<|end_message|>     (if tools)
    <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>
    <|message_user|><|content_text|>...<|end_message|>
    <|message_model|><|content_thinking|>...<|end_message|>                 (reasoning channel)
    <|message_model|><|content_text|>...<|end_message|><|content_model_end_sampling|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":..}<|end_message|>
    <|message_tool|>NAME<|content_text|>OBS<|end_message|>

We build the token stream by *incremental rendering*: render the message list up to
turn i and up to turn i+1, and take the token diff as turn i+1's span. That way the
template — not us — owns the exact byte format, and the loss mask lands on real spans.
"""
import argparse, json, os, re, sys

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def split_think(v: str):
    m = THINK.search(v)
    if not m:
        return None, v.strip()
    return m.group(1).strip(), v[m.end():].strip()


TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_call(v: str):
    """function_call turn -> (think, visible_text, [ {name, arguments}, ... ]).

    Body after </think> is: optional visible prose, then one or more
    <tool_call>{"name":..,"arguments":{..}}</tool_call> blocks.
    """
    think, body = split_think(v)
    calls = []
    for m in TOOLCALL.finditer(body):
        try:
            d = json.loads(m.group(1))
        except Exception:
            return think, None, None
        name = d.get("name") or d.get("action")
        if not name:
            return think, None, None
        args = d.get("arguments", d.get("args", d.get("action_input", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}
        calls.append({"name": name, "arguments": args})
    if not calls:
        return think, None, None
    visible = TOOLCALL.sub("", body).strip()
    return think, visible, calls


def to_oai(row):
    """LLaMA-Factory row -> (list of openai-ish messages, list of per-message train flag)."""
    msgs, flags = [], []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
        flags.append(False)
    pending_name = None
    for t in row["conversations"]:
        fr, v, loss = t["from"], t["value"], bool(t.get("loss"))
        if fr == "human":
            msgs.append({"role": "user", "content": v}); flags.append(False)
        elif fr == "gpt":
            think, ans = split_think(v)
            m = {"role": "assistant", "content": ans}
            if think:
                m["reasoning_content"] = think
            msgs.append(m); flags.append(loss)
        elif fr == "function_call":
            think, visible, calls = parse_tool_call(v)
            if calls is None:
                return None, None            # unparseable -> drop the row
            m = {"role": "assistant", "content": visible or "",
                 "tool_calls": [{"id": f"c{len(msgs)}_{j}", "function": c}
                                for j, c in enumerate(calls)]}
            if think:
                m["reasoning_content"] = think
            msgs.append(m); flags.append(loss)
            pending_name = calls[0]["name"]
        elif fr == "observation":
            msgs.append({"role": "tool", "name": pending_name or "tool", "content": v})
            flags.append(False)
        else:
            return None, None
    return msgs, flags


def render_spans(tok, msgs, flags, tools, effort=0.9):
    """Incrementally render; return (all_ids, weights) with weight 1.0 on trained turns."""
    kw = dict(tokenize=False, reasoning_effort=effort)
    if tools:
        kw["tools"] = tools
    ids, weights, prev_text, prev_ids = [], [], "", []
    for i in range(1, len(msgs) + 1):
        text = tok.apply_chat_template(msgs[:i], **kw)
        if not text.startswith(prev_text):
            return None, None                # non-monotonic render -> drop
        cur_ids = tok.encode(text)
        # the template is a pure concatenation, so re-tokenising the delta is safe and
        # avoids trusting that prev_ids is a token-prefix of cur_ids
        delta_ids = cur_ids[len(prev_ids):]
        if len(delta_ids) <= 0:
            return None, None
        w = 1.0 if flags[i - 1] else 0.0
        ids.extend(delta_ids); weights.extend([w] * len(delta_ids))
        prev_text, prev_ids = text, cur_ids
    return ids, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--holdout", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    src = [json.loads(l) for l in open(a.src) if l.strip()]
    if a.limit:
        src = src[: a.limit]
    print(f"[build] {len(src)} source rows", file=sys.stderr, flush=True)

    kept = []
    drop = {"unparseable": 0, "too_long": 0, "render": 0, "no_trained_tokens": 0}
    for i, r in enumerate(src):
        msgs, flags = to_oai(r)
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
            drop["no_trained_tokens"] += 1; continue
        kept.append({"id": f"{i:05d}", "ids": ids, "w": w,
                     "n": len(ids), "n_train": int(sum(w)),
                     "kind": "agentic" if tools else "plain",
                     # keep the prompt-only message list so we can re-sample later
                     "msgs_prompt": [m for m in msgs if m["role"] in ("system", "user")][:2],
                     "tools": r.get("tools") or ""})
        if (i + 1) % 200 == 0:
            print(f"[build] {i+1}/{len(src)} kept={len(kept)} {drop}", file=sys.stderr, flush=True)

    print(f"[build] kept {len(kept)}  dropped {drop}", file=sys.stderr)

    import random
    rng = random.Random(a.seed); rng.shuffle(kept)
    hold, train = kept[: a.holdout], kept[a.holdout:]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for name, part in (("train", train), ("holdout", hold)):
        p = a.out.replace(".jsonl", f".{name}.jsonl")
        with open(p, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        tot = sum(r["n"] for r in part); tr = sum(r["n_train"] for r in part)
        nag = sum(r["kind"] == "agentic" for r in part)
        print(f"[build] {p}: {len(part)} rows ({nag} agentic), "
              f"{tot/1e6:.2f}M tok, {tr/1e6:.2f}M trained", file=sys.stderr)


if __name__ == "__main__":
    main()

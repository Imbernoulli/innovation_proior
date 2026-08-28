"""Regenerate the <think> of every trained assistant turn with the Tinker-tuned Inkling.

Why only the think: the complaint is that our hand-written reasoning is off-policy.
The answers / tool calls are the actual scientific content and the recorded
observations are only valid for the recorded actions, so both stay verbatim.
Each turn is sampled teacher-forced on the real prefix.

Output is a LLaMA-Factory jsonl, byte-identical to the source except that each
trained turn's <think>...</think> is replaced by the model's own reasoning.
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import tinker
from tinker import types as tt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import to_oai, THINK

STOP = ["<|end_message|>"]


CONDITION_HINT = (
    "You already know where this turn lands. Below is the conclusion you reached. "
    "Write the reasoning that actually gets there — your own line of attack, in your "
    "own voice. Do not announce the conclusion up front and do not summarise it; "
    "think your way to it, including the parts you had to rule out.\n\n"
    "--- the conclusion this reasoning must reach ---\n{answer}"
)


def prompt_for_turn(tok, msgs, i, tools, effort=0.9, answer=None):
    """Text prompt that ends exactly where the model should start thinking.

    With `answer`, an extra system turn shows the teacher where this turn lands
    (arm B). Answer-blind sampling (arm A) produces reasoning that is honest but
    systematically converges less onto the answer it precedes — measured at
    0.751 -> 0.488 answer-term reach, a drop on 22 of 24 gate turns — which would
    train the student on a non-sequitur. The conditioning turn exists only in the
    prompt; it is never written back into the corpus.
    """
    kw = dict(tokenize=False, reasoning_effort=effort)
    if tools:
        kw["tools"] = tools
    seq = list(msgs[:i])
    if answer:
        seq = seq + [{"role": "system", "content": CONDITION_HINT.format(answer=answer)}]
    head = tok.apply_chat_template(seq, add_generation_prompt=True, **kw)
    return head + "<|content_thinking|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--state", default=".cache/tinker/inkling_run.json")
    ap.add_argument("--model-path", default=None, help="override the tinker checkpoint path")
    ap.add_argument("--base-model", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--tokenizer", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--out", default=".cache/tinker/innovation_distilled.jsonl")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-prompt", type=int, default=98304)
    ap.add_argument("--condition-on-answer", action="store_true",
                    help="arm B: show the teacher the answer this turn must reach")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    model_path = a.model_path
    if model_path is None:
        st = json.load(open(a.state))
        model_path = st.get("sampler_id") or st.get("model_path")
    print(f"[sample] model {model_path}", flush=True)

    sc = tinker.ServiceClient()
    cl = sc.create_sampling_client(model_path=model_path)

    rows = [json.loads(l) for l in open(a.src) if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    # resume: keep whatever ids already landed
    done = {}
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except Exception:
                continue
            done[r["_id"]] = r
    print(f"[sample] {len(rows)} rows, {len(done)} already done", flush=True)

    lock = threading.Lock()
    fout = open(a.out, "a")
    counters = {"rows": 0, "turns": 0, "fail": 0, "trunc": 0, "t0": time.time()}

    def do_row(idx_row):
        idx, r = idx_row
        rid = f"{idx:05d}"
        if rid in done:
            return
        msgs, flags = to_oai(r)
        if msgs is None:
            return
        tools = None
        if (r.get("tools") or "").strip():
            try:
                tools = json.loads(r["tools"])
            except Exception:
                return
        out = json.loads(json.dumps(r))          # deep copy
        # map: index into `msgs` -> index into conversations
        conv_idx = [j for j, t in enumerate(out["conversations"])]
        # msgs may carry a leading system message that conversations does not
        off = 1 if (r.get("system") and msgs and msgs[0]["role"] == "system") else 0
        n_new = 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or not flags[i]:
                continue
            cj = conv_idx[i - off]
            src_val = out["conversations"][cj]["value"]
            if not THINK.search(src_val):
                continue
            try:
                cond = None
                if a.condition_on_answer:
                    om = THINK.search(src_val)
                    cond = src_val[om.end():].strip() if om else None
                ptxt = prompt_for_turn(tok, msgs, i, tools, answer=cond)
                pids = tok.encode(ptxt)
                if len(pids) > a.max_prompt:
                    continue
                resp = cl.sample(
                    prompt=tt.ModelInput.from_ints(pids),
                    num_samples=1,
                    sampling_params=tt.SamplingParams(
                        max_tokens=a.max_tokens, temperature=a.temperature,
                        top_p=a.top_p, stop=STOP),
                ).result()
                seq = resp.sequences[0]
                text = tok.decode(seq.tokens)
                for s in STOP:
                    text = text.split(s)[0]
                think = text.strip()
                if len(think) < 200:
                    with lock:
                        counters["fail"] += 1
                    continue
                if getattr(seq, "stop_reason", None) == "length":
                    with lock:
                        counters["trunc"] += 1
                out["conversations"][cj]["value"] = THINK.sub(
                    lambda _: f"<think>\n{think}\n</think>", src_val, count=1)
                n_new += 1
            except Exception as e:
                with lock:
                    counters["fail"] += 1
                    if counters["fail"] <= 5:
                        print(f"[sample] turn fail {rid}/{i}: {type(e).__name__} {e}", flush=True)
        if n_new == 0:
            return
        out["_id"] = rid
        out["_n_regen"] = n_new
        with lock:
            fout.write(json.dumps(out, ensure_ascii=False) + "\n"); fout.flush()
            counters["rows"] += 1; counters["turns"] += n_new
            if counters["rows"] % 20 == 0:
                el = time.time() - counters["t0"]
                print(f"[sample] rows {counters['rows']} turns {counters['turns']} "
                      f"fail {counters['fail']} trunc {counters['trunc']} {el:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(do_row, list(enumerate(rows))))
    print(f"[sample] done {counters}", flush=True)


if __name__ == "__main__":
    main()

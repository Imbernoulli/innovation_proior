"""Regenerate the <think> of every trained assistant turn with the Tinker-tuned Qwen3.8.

Same contract as sample_inkling.py -- only the think is rewritten, the answer and
tool calls stay verbatim, and each turn is sampled teacher-forced on the real
prefix -- but the wire format is Qwen's, not Inkling's:

  * the generation prompt already ends with "<|im_start|>assistant\\n<think>\\n",
    so sampling starts inside the think and we stop at "</think>"
  * the think must be passed as `reasoning_content`, never left inside `content`,
    or the template emits an empty think followed by the real one

`--condition-on-answer` and `--method-name` are the two conditioning variants:
the first shows the teacher the whole answer, the second only a short name for
where the turn lands. Either way the extra turn lives in the prompt only and is
never written back into the corpus.
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import tinker
from tinker import types as tt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import THINK
from build_data_qwen import to_qwen

STOP = ["</think>"]

COND_ANSWER = (
    "You already know where this turn lands. Below is the conclusion you reached. "
    "Write the reasoning that actually gets there — your own line of attack, in your "
    "own voice. Do not announce the conclusion up front and do not summarise it; "
    "think your way to it, including the parts you had to rule out.\n\n"
    "--- the conclusion this reasoning must reach ---\n{answer}"
)
COND_NAME = (
    "The method this turn lands on is: {name}\n\n"
    "Write the reasoning that gets there from the question — your own line of attack, "
    "in your own voice. Do not announce it up front; derive it, including what you "
    "had to rule out along the way."
)


def prompt_for_turn(tok, msgs, i, tools, cond=None):
    kw = {}
    if tools:
        kw["tools"] = tools
    seq = list(msgs[:i])
    if cond:
        seq = seq + [{"role": "system", "content": cond}]
    return tok.apply_chat_template(seq, tokenize=False, add_generation_prompt=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--state", default=".cache/tinker/qwen38_run.json")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default=".cache/tinker/innovation_distilled_q38.jsonl")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-prompt", type=int, default=57344)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--condition-on-answer", action="store_true")
    ap.add_argument("--method-name-field", default=None,
                    help="json file mapping row id -> short method name (arm C)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    model_path = a.model_path or json.load(open(a.state)).get("model_path")
    print(f"[sample] model {model_path}", flush=True)
    sc = tinker.ServiceClient()
    cl = sc.create_sampling_client(model_path=model_path)

    names = json.load(open(a.method_name_field)) if a.method_name_field else {}
    rows = [json.loads(l) for l in open(a.src) if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["_id"])
            except Exception:
                pass
    print(f"[sample] {len(rows)} rows, {len(done)} already done", flush=True)

    lock = threading.Lock(); fout = open(a.out, "a")
    c = {"rows": 0, "turns": 0, "fail": 0, "trunc": 0, "t0": time.time()}

    def do_row(ir):
        idx, r = ir
        rid = f"{idx:05d}"
        if rid in done:
            return
        msgs, flags = to_qwen(r)
        if msgs is None:
            return
        tools = None
        if (r.get("tools") or "").strip():
            try:
                tools = json.loads(r["tools"])
            except Exception:
                return
        out = json.loads(json.dumps(r))
        off = 1 if (r.get("system") and msgs and msgs[0]["role"] == "system") else 0
        n_new = 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or not flags[i]:
                continue
            cj = i - off
            src_val = out["conversations"][cj]["value"]
            om = THINK.search(src_val)
            if not om:
                continue
            cond = None
            if a.condition_on_answer:
                cond = COND_ANSWER.format(answer=src_val[om.end():].strip())
            elif names.get(rid):
                cond = COND_NAME.format(name=names[rid])
            try:
                pids = tok.encode(prompt_for_turn(tok, msgs, i, tools, cond))
                if len(pids) > a.max_prompt:
                    continue
                resp = cl.sample(prompt=tt.ModelInput.from_ints(pids), num_samples=1,
                                 sampling_params=tt.SamplingParams(
                                     max_tokens=a.max_tokens, temperature=a.temperature,
                                     top_p=a.top_p, stop=STOP)).result()
                seq = resp.sequences[0]
                text = tok.decode(seq.tokens)
                for s in STOP:
                    text = text.split(s)[0]
                think = text.strip()
                if len(think) < a.min_chars:
                    with lock:
                        c["fail"] += 1
                    continue
                if getattr(seq, "stop_reason", None) == "length":
                    with lock:
                        c["trunc"] += 1
                out["conversations"][cj]["value"] = THINK.sub(
                    lambda _: f"<think>\n{think}\n</think>", src_val, count=1)
                n_new += 1
            except Exception as e:
                with lock:
                    c["fail"] += 1
                    if c["fail"] <= 5:
                        print(f"[sample] turn fail {rid}/{i}: {type(e).__name__} {e}", flush=True)
        if n_new == 0:
            return
        out["_id"] = rid; out["_n_regen"] = n_new
        with lock:
            fout.write(json.dumps(out, ensure_ascii=False) + "\n"); fout.flush()
            c["rows"] += 1; c["turns"] += n_new
            if c["rows"] % 50 == 0:
                print(f"[sample] rows {c['rows']} turns {c['turns']} fail {c['fail']} "
                      f"trunc {c['trunc']} {time.time()-c['t0']:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(do_row, list(enumerate(rows))))
    print(f"[sample] done {c}", flush=True)


if __name__ == "__main__":
    main()

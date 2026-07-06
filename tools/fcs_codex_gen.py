#!/usr/bin/env python3
"""Build FrontierCS-style coding SFT data by BLACK-BOX calling Codex (`codex exec`).

For each hard problem in a list that mimics the FrontierCS distribution
(algorithm C++ / research-cpu Python / research-gpu Triton), invoke Codex once and
let it produce the whole training example: CONTEXT + REASONING + ANSWER.

Codex runs headless (`codex exec`, model gpt-5.5) in a per-problem scratch cwd with
workspace-write sandbox, so it can COMPILE/RUN to self-verify before landing.

Landing format per category (this is the whole point — FrontierCS only scores the landing):
  algorithm    -> ONE self-contained C++17 program reading stdin, one ```cpp block
  research_cpu -> Python implementing the given API class/functions (CPU, no GPU)
  research_gpu -> Python + Triton kernel implementing the given API (GPU)

Reasoning must carry the FrontierCS disposition the eval rewards: complexity/TLE
awareness, a real worked-example verification, general algorithm (NO hardcoding),
and a fallback to the simplest correct solution when a clever idea is not clearly safe.

Output: out_dir/<id>/{context.md, reasoning.md, answer.md, meta.json}. Resumable.
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

LANDING = {
    'algorithm': ("ONE self-contained C++17 program that reads from standard input and writes to "
                  "standard output, as a single ```cpp code block. No prose outside the code block."),
    'research_cpu': ("Python code implementing EXACTLY the class/functions named in the problem's API "
                     "section (CPU-only, 8 vCPU / 16GB, no GPU), as a single ```python code block. "
                     "The final class name must match the API spec exactly."),
    'research_gpu': ("Python code using Triton (triton 3.2.0 / CUDA) implementing EXACTLY the API named "
                     "in the problem, targeting an NVIDIA L4 (24GB), as a single ```python code block."),
}

PROMPT_TMPL = """You are producing ONE supervised-training example for a hard problem. Do NOT ask questions and do NOT modify anything outside your scratch working directory. You MAY compile/run code to verify your solution. Output plain text to stdout with EXACTLY these three delimited sections and NOTHING else before/after:

===CONTEXT===
A concise, self-contained statement of the problem, its constraints, and its scoring, written as a contemporaneous framing (what we are given and must produce). No meta commentary, no mention of this being a training example.

===REASONING===
Your full first-person solving trace. REQUIRED disposition: (1) state the input scale and the complexity budget, and explicitly reject approaches that would TLE/MLE; (2) design a GENERAL algorithm — never hardcode per-case constants; (3) VERIFY on a concrete worked example (hand-trace a small instance and check the output; sanity-check against brute force where feasible); (4) if a clever construction is not clearly correct within budget, FALL BACK to the simplest correct solution and land THAT. Keep going until you have one general, correct, efficient solution. Write it as an earned narrative, not assertions like "this is guaranteed".

===ANSWER===
{landing}

The problem:

{statement}
"""


def build_prompt(prob):
    cat = prob.get('category', 'algorithm')
    return PROMPT_TMPL.format(landing=LANDING.get(cat, LANDING['algorithm']),
                              statement=prob['statement'].strip())


def parse_sections(text):
    out = {}
    for name in ('CONTEXT', 'REASONING', 'ANSWER'):
        m = re.search(rf'==={name}===\s*(.*?)(?=\n===(?:CONTEXT|REASONING|ANSWER)===|\Z)',
                      text, re.S)
        out[name.lower()] = (m.group(1).strip() if m else '')
    return out


def extract_code(answer, cat):
    lang = 'cpp' if cat == 'algorithm' else 'python'
    m = re.findall(rf'```(?:{lang})?\s*\n(.*?)```', answer, re.S)
    return (m[-1].strip() if m else '')


def run_codex(prob, args):
    pid = prob['id']
    cat = prob.get('category', 'algorithm')
    scratch = os.path.join(args.scratch, pid)
    os.makedirs(scratch, exist_ok=True)
    prompt = build_prompt(prob)
    t0 = time.time()
    try:
        p = subprocess.run(
            ['codex', 'exec', '--skip-git-repo-check',
             '-c', f'model_reasoning_effort="{args.effort}"', prompt],
            cwd=scratch, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=args.timeout)
        out = p.stdout or ''
    except subprocess.TimeoutExpired:
        return {'id': pid, 'ok': False, 'error': 'codex-timeout', 'secs': round(time.time() - t0)}
    except Exception as e:
        return {'id': pid, 'ok': False, 'error': f'{type(e).__name__}: {e}'}
    secs = round(time.time() - t0)
    sec = parse_sections(out)
    code = extract_code(sec['answer'], cat)
    ok = bool(sec['context'] and sec['reasoning'] and sec['answer'] and code)
    d = os.path.join(args.out, pid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'context.md'), 'w') as f: f.write(sec['context'])
    with open(os.path.join(d, 'reasoning.md'), 'w') as f: f.write(sec['reasoning'])
    with open(os.path.join(d, 'answer.md'), 'w') as f: f.write(sec['answer'])
    with open(os.path.join(d, '_codex_raw.txt'), 'w') as f: f.write(out)
    meta = {'id': pid, 'category': cat, 'ok': ok, 'secs': secs,
            'code_chars': len(code), 'reasoning_chars': len(sec['reasoning']),
            'source': 'codex:gpt-5.5'}
    with open(os.path.join(d, 'meta.json'), 'w') as f: json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', default='data_v4/_fcs_codex/problems.jsonl')
    ap.add_argument('--out', default='data_v4/_fcs_codex/gen')
    ap.add_argument('--scratch', default=os.environ.get('TMPDIR', '/tmp') + '/fcs_codex_scratch')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--effort', default='high', help='codex reasoning effort (low..xhigh)')
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    probs = [json.loads(l) for l in open(args.problems) if l.strip()]
    todo = [p for p in probs if not os.path.exists(os.path.join(args.out, p['id'], 'meta.json'))]
    if args.limit:
        todo = todo[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f'{len(probs)} problems, {len(todo)} to do (skip {len(probs)-len(todo)} done); '
          f'concurrency={args.concurrency} effort={args.effort}', flush=True)
    done = ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_codex, p, args): p['id'] for p in todo}
        for fu in as_completed(futs):
            r = fu.result(); done += 1; ok += bool(r.get('ok'))
            print(f'  [{done}/{len(todo)}] {r["id"]:22} ok={r.get("ok")} '
                  f'{r.get("secs","?")}s code={r.get("code_chars","-")} err={r.get("error","")}',
                  flush=True)
    print(f'DONE: {ok}/{done} complete data points', flush=True)


if __name__ == '__main__':
    main()

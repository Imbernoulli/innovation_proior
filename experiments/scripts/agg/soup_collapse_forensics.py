#!/usr/bin/env python3
"""Forensics: why does pure SFT (alpha=1.0) collapse on FCS vs base?

Uses the OFFICIAL extraction chain: strip_think (everything through LAST </think>)
-> official extract_cpp_code (longest ```cpp fence, fallback = whole text).
Then compiles each extracted submission with g++ -fsyntax-only to count real
compile errors, and buckets every zero-score sample into ONE mechanism category.
"""
import glob, json, os, re, subprocess, sys, tempfile, zlib
from concurrent.futures import ThreadPoolExecutor

OUT = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs"

def strip_think(response):
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response

CPP_FENCE = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)

def extract_cpp_code(response_text):
    if not response_text:
        return "", False
    code = response_text.strip()
    matches = CPP_FENCE.findall(code)
    if matches:
        return max(matches, key=len).strip(), True
    if code.startswith("```cpp"):
        code = code[6:].strip()
    elif code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code, False

def load_fcs(tag):
    seen = {}
    for f in sorted(glob.glob(os.path.join(OUT, f"cc_eval_{tag}_thinking_32k_both_vllm", "shard_*", "samples.jsonl"))):
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d["data_source"] != "frontiercs":
                continue
            seen[(str(d["problem_idx"]), str(d["sample_idx"]))] = d
    return seen

def compiles(code, timeout=45):
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "-w", path],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr or "")[:400]
    except subprocess.TimeoutExpired:
        return False, "COMPILE_TIMEOUT"
    finally:
        os.unlink(path)

def looks_python(code):
    if "#include" in code or "int main" in code:
        return False
    return bool(re.search(r"^\s*(def |import |from \w+ import|print\()", code, re.M))

def degenerate(text):
    if len(text) < 2000:
        return False
    tail = text[-20000:].encode(errors="replace")
    ratio = len(zlib.compress(tail, 6)) / len(tail)
    return ratio < 0.05

def analyze(tag, do_compile=True):
    rows = load_fcs(tag)
    recs = []
    def work(item):
        (pid, sid), d = item
        text = d.get("text") or ""
        think = text.rpartition("</think>")[0] if "</think>" in text else text
        stripped = strip_think(text)
        code, fenced = extract_cpp_code(stripped)
        rec = {
            "pid": pid, "sid": sid,
            "score": float(d["metrics"].get("reward") or 0.0),
            "error": bool(d.get("error")),
            "tokens": d.get("completion_tokens") or 0,
            "trunc": (d.get("completion_tokens") or 0) >= 32768,
            "closed_think": "</think>" in text,
            "think_chars": len(think),
            "ans_chars": len(stripped) if "</think>" in text else 0,
            "code_chars": len(code),
            "fenced": fenced,
            "no_code": len(code.strip()) == 0,
            "python": looks_python(code) if code else False,
            "degen": degenerate(text),
        }
        if do_compile and code and not rec["python"]:
            ok, err = compiles(code)
            rec["compile_ok"] = ok
            rec["compile_err"] = "" if ok else err
        else:
            rec["compile_ok"] = None
        return rec
    with ThreadPoolExecutor(max_workers=12) as ex:
        recs = list(ex.map(work, rows.items()))
    return recs

def bucket(r):
    """mutually exclusive mechanism bucket for a ZERO-score sample"""
    if r["error"]:
        return "infra_error"
    if r["no_code"]:
        return "no_code_extracted"
    if r["python"]:
        return "wrong_language_python"
    if r["trunc"] and not r["closed_think"] and not r["fenced"]:
        return "truncation_no_submission"
    if r["compile_ok"] is False:
        return "compile_error"
    if r["degen"]:
        return "degenerate_repetition"
    return "compiles_but_scores_zero"  # wrong algorithm / WA / TLE / runtime

def summarize(tag, recs):
    n = len(recs)
    zero = [r for r in recs if r["score"] <= 0]
    pos = [r for r in recs if r["score"] > 0]
    from collections import Counter
    buckets = Counter(bucket(r) for r in zero)
    comp_attempted = [r for r in recs if r["compile_ok"] is not None]
    print(f"\n===== {tag}  (n={n}) =====")
    print(f" score>0: {len(pos)} ({len(pos)/n:.1%});  mean sample score {sum(r['score'] for r in recs)/n:.2f}")
    print(f" closed_think: {sum(r['closed_think'] for r in recs)/n:.1%}  trunc: {sum(r['trunc'] for r in recs)/n:.1%}"
          f"  fenced-code: {sum(r['fenced'] for r in recs)/n:.1%}  no_code: {sum(r['no_code'] for r in recs)/n:.1%}")
    tl = sorted(r["think_chars"] for r in recs)
    tk = sorted(r["tokens"] for r in recs)
    print(f" think_chars median {tl[n//2]}, mean {sum(tl)/n:.0f};  completion_tokens median {tk[n//2]}")
    if comp_attempted:
        ok = sum(1 for r in comp_attempted if r["compile_ok"])
        print(f" compile pass rate (of extracted C++ submissions): {ok}/{len(comp_attempted)} = {ok/len(comp_attempted):.1%}")
    print(f" ZERO-SCORE buckets (n={len(zero)}):")
    for k, v in buckets.most_common():
        print(f"   {k:28s} {v:4d}  ({v/len(zero):.1%})")
    return recs

if __name__ == "__main__":
    tags = sys.argv[1:] or ["clean_start", "clean_clean_nomaintain_wd01_sft", "clean_clean_full_wd01_sft",
                            "clean_clean_nomaintain_wd01_a10", "clean_clean_nomaintain_wd01_a50"]
    allrecs = {}
    for t in tags:
        allrecs[t] = summarize(t, analyze(t))
    json.dump({t: rs for t, rs in allrecs.items()},
              open("collapse_forensics.json", "w"))

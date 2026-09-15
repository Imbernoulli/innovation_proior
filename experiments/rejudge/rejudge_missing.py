#!/usr/bin/env python3
"""Re-score the cells that have a stored generation but no score.

Every "missing" cell in the cc_eval_* outputs is a JUDGE-side failure: the model's `text` is
stored, only `metrics.score` is absent (the row carries `error` instead).  So the hole can be
filled without re-generating -- which also keeps the generation byte-identical, something a
fresh eval could never do (vLLM sampling is not reproducible).

Covers all three data_sources (the pre-existing rejudge_samples.py did frontiercs only, and
explicitly SKIPPED error rows, i.e. exactly the rows we need):
  frontiercs           needs the node judge   (FRONTIERCS_JUDGE_URL)
  alebench             needs ALE_BENCH_* + apptainer; fresh session per row
  frontiercs_research  needs the research overlay python + a WRITABLE problem tree

Why a writable tree: the research evaluators write next to their inputs, and the official cache
plus the julia env live in a read-only tree, so they died with EACCES:
  symbolic_regression/*      -> julia_env/lock.pid            (PySR/juliacall lock)
  grammar_fuzzing/fuzzer/sql -> problems/.../sql/output_ans    (evaluator writes its answer file)
REJUDGE_ENV points at our own copy; OFFICIAL_ROOT is monkey-patched onto it (the driver derives
it from __file__, there is no env var).

Output is written as a normal shard directory (`shard_rejudge`) next to the originals.  The
dedupe rule downstream is "last row in sorted glob order wins", and `shard_rejudge` sorts after
`shard_0`/`shard_1`, so the repaired rows take over automatically.  NOTHING is overwritten.

  python rejudge_missing.py --arm base9b_v2c --bench frontiercs --keys rejudge_keys.json
"""
import argparse, collections, glob, json, os, sys, threading, time
import concurrent.futures as cf

FS = "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
REJUDGE_ENV = os.environ.get("REJUDGE_ENV", f"{D}/rejudge_env")
SUB = {"frontiercs": "thinking_32k_both_vllm", "alebench": "thinking_32k_both_vllm",
       "frontiercs_research": "research_thinking_32k_vllm"}


def _patch_research():
    """Point the research evaluators at our writable copy of the official problem tree."""
    from pathlib import Path
    root = Path(REJUDGE_ENV) / "Frontier-CS-official"
    if not (root / "research" / "problems").is_dir():
        raise SystemExit(f"[rejudge] writable research tree missing: {root}")
    import frontiercs_research_eval as fre
    mods = [fre]
    try:
        import frontiercs_research_cpu_eval as frc
        mods.append(frc)
    except Exception:
        pass
    for m in mods:
        m.OFFICIAL_ROOT = root
        m.RESEARCH_ROOT = root / "research"
        if hasattr(m, "RESEARCH_PROBLEMS"):
            m.RESEARCH_PROBLEMS = root / "research" / "problems"
    print(f"[rejudge] research OFFICIAL_ROOT -> {root}", flush=True)


def load_rows(arm, bench, want):
    """Last stored row per wanted key, preferring one that actually has text."""
    best = {}
    for f in sorted(glob.glob(f"{D}/outputs/cc_eval_{arm}_{SUB[bench]}/shard_*/samples.jsonl")):
        if "/shard_rejudge/" in f:
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("data_source") != bench:
                continue
            k = (str(r["ground_truth"]), int(r.get("sample_idx", -1)))
            if k not in want:
                continue
            if (r.get("text") or "") or k not in best:
                best[k] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bench", required=True, choices=list(SUB))
    ap.add_argument("--keys", required=True, help="json: {'<arm>|<bench>': [[gt, sample_idx], ...]}")
    ap.add_argument("--judge-url", default=os.environ.get("FRONTIERCS_JUDGE_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("REJUDGE_WORKERS", "4")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sys.path.insert(0, f"{FS}/scripts")
    from eval_qwen35_base_vllm_request import _score
    if a.bench == "frontiercs_research":
        _patch_research()
    if a.bench == "frontiercs":
        # The 154 lost FrontierCS cells are all "Evaluation timed out after 1000s" -- that is the
        # runner's POLL timeout, not a verdict. Raise it so the heavy problems (143/148/149/153/
        # 160/169) get to finish; DEFAULT_TIMEOUT is a plain class attribute.
        import eval_qwen35_base_vllm_request as drv
        drv.AlgorithmicLocalRunner.DEFAULT_TIMEOUT = int(os.environ.get("REJUDGE_FCS_TIMEOUT", "3000"))
        print(f"[rejudge] FCS poll timeout -> {drv.AlgorithmicLocalRunner.DEFAULT_TIMEOUT}s", flush=True)

    want = {(str(g), int(i)) for g, i in json.load(open(a.keys)).get(f"{a.arm}|{a.bench}", [])}
    if not want:
        print(f"[rejudge] nothing to do for {a.arm}|{a.bench}"); return 0
    rows = load_rows(a.arm, a.bench, want)
    todo = [(k, r) for k, r in rows.items() if (r.get("text") or "")]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[rejudge] {a.arm} {a.bench}: {len(want)} wanted, {len(todo)} have text", flush=True)

    out = a.out or f"{D}/outputs/cc_eval_{a.arm}_{SUB[a.bench]}/shard_rejudge/samples.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done, lock, t0 = [0], threading.Lock(), time.time()
    tally = collections.Counter()

    def work(item):
        k, r = item
        rec = {kk: r.get(kk) for kk in ("data_source", "ground_truth", "problem_idx", "sample_idx",
                                        "completion_tokens", "prompt_variant", "score_backend",
                                        "request_seed", "text")}
        rec["rejudged"] = True
        try:
            m = _score(a.bench, r["text"], str(r["ground_truth"]), a.judge_url,
                       frontiercs_score_backend="official")
            rec["metrics"] = {kk: vv for kk, vv in m.items()}
            rec["error"] = None
            tally["scored"] += 1
            tally["nonzero"] += 1 if (m.get("score") or 0) > 0 else 0
        except Exception as exc:                      # still infra-broken -> keep it an error row
            rec["metrics"] = {"reward": 0.0, "score": 0.0}
            rec["error"] = repr(exc)[:400]
            rec["prev_error"] = str(r.get("error"))[:200]
            tally["still_failed"] += 1
        with lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(todo):
                print(f"[rejudge] {done[0]}/{len(todo)} ok={tally['scored']} "
                      f"fail={tally['still_failed']} {time.time()-t0:.0f}s", flush=True)
        return rec

    with open(out, "w") as fh, cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for rec in ex.map(work, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"[rejudge] DONE {a.arm} {a.bench} -> {out}  {dict(tally)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Correctness guard for the decoupled eval path (--decouple-scoring).
#
# The decoupled loop must produce records IDENTICAL in shape and seed to the
# coupled `_run_one` path -- only the SCHEDULING differs (generation is never
# blocked on the judge). These tests stub out generation and scoring so no model
# or judge is needed, then assert:
#   1. same request_seed for a given (problem_idx, sample_idx)
#   2. same record keys and same metric values
#   3. generation is NOT serialized behind scoring (a slow judge does not stop
#      new generations from starting) -- the whole point of the change
#   4. preflight refuses iterative-refinement mode (which cannot be decoupled)
#
# Run: python -m pytest verl/tests/trainer/ppo/test_decoupled_eval_on_cpu.py -q

import importlib.util
import queue
import threading
import time
import types
from pathlib import Path

import pytest


def _load_eval_module():
    path = Path(__file__).resolve().parents[4] / "scripts" / "eval_qwen35_base_vllm_request.py"
    spec = importlib.util.spec_from_file_location("eval_qwen35_base_vllm_request", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_eval_module()


def _args(**kw):
    a = types.SimpleNamespace(
        model="m", temperature=1.0, top_p=0.95, max_tokens=100, presence_penalty=1.5,
        frequency_penalty=None, top_k=20, min_p=0.0, repetition_penalty=None,
        enable_thinking=True, base_url="http://x", timeout=10, seed=42, n_samples=2,
        judge_url="http://judge", frontiercs_score_backend="official",
        frontiercs_iterative_rounds=kw.pop("frontiercs_iterative_rounds", 1),
        save_text=False, text_preview_chars=0,
        concurrency=kw.pop("concurrency", 4), score_concurrency=kw.pop("score_concurrency", 2),
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _problems(n):
    return [
        {"data_source": "frontiercs", "ground_truth": f"p{i}", "prompt_variant": "frontiercs:official-generate_solutions",
         "messages": [{"role": "user", "content": f"problem {i}"}]}
        for i in range(n)
    ]


def _patch_gen_score(monkeypatch, gen_delay=0.0, score_delay=0.0):
    monkeypatch.setattr(ev, "_generate_one",
        lambda args, messages, seed: (time.sleep(gen_delay), f"text-seed-{seed}", 7, gen_delay)[1:])
    monkeypatch.setattr(ev, "_score",
        lambda ds, text, gt, judge_url, *, frontiercs_score_backend: (time.sleep(score_delay), {"reward": 1.0, "score": 1.0, "score_unbounded": 1.0})[1])


def test_record_shape_and_seed_match_coupled(monkeypatch):
    """Decoupled record must equal the coupled _run_one record (keys, seed, metrics)."""
    _patch_gen_score(monkeypatch)
    args = _args()
    prob = _problems(1)[0]

    # coupled reference
    coupled = ev._run_one(args, prob, 0, 1)

    # decoupled
    gen_q, result_q = queue.Queue(), queue.Queue()
    gen_active, lock = [1], threading.Lock()
    scorer = threading.Thread(target=ev._decoupled_score_worker, args=(args, gen_q, result_q), daemon=True)
    scorer.start()
    ev._decoupled_gen_worker(args, prob, 0, 1,
                             None if args.seed is None else args.seed + 0 * args.n_samples + 1,
                             gen_q, gen_active, lock)
    rec = result_q.get(timeout=5)
    assert set(rec) == set(coupled), f"key mismatch: {set(rec) ^ set(coupled)}"
    assert rec["request_seed"] == coupled["request_seed"]
    assert rec["metrics"] == coupled["metrics"]
    assert rec["data_source"] == coupled["data_source"] and rec["ground_truth"] == coupled["ground_truth"]


def test_generation_not_blocked_by_scoring(monkeypatch):
    """The fix: a SLOW judge must not stop new generations from being produced.
    With a slow scorer, generations should all finish quickly while scores lag."""
    _patch_gen_score(monkeypatch, gen_delay=0.02, score_delay=0.3)
    args = _args(concurrency=4, score_concurrency=1)
    probs = _problems(6)
    tasks = [(i, p, 0) for i, p in enumerate(probs)]

    import io
    records, planned = {}, set()
    gen_q, result_q = queue.Queue(), queue.Queue()
    gen_active, lock = [len(tasks)], threading.Lock()
    scorer = threading.Thread(target=ev._decoupled_score_worker, args=(args, gen_q, result_q), daemon=True)
    scorer.start()

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(ev._decoupled_gen_worker, args, p, i, s, 42 + i * args.n_samples + s,
                          gen_q, gen_active, lock) for i, p, s in tasks]
        for f in futs:
            f.result()
    gen_done = time.time() - t0
    # 6 generations at 0.02s with 4 workers should take ~0.04s, NOT 6*0.3s of scoring.
    assert gen_done < 0.3, f"generation was blocked by scoring (took {gen_done:.2f}s)"
    # drain
    seen = 0
    while seen < len(tasks):
        result_q.get(timeout=5); seen += 1


def test_preflight_refuses_iterative(monkeypatch):
    args = _args(frontiercs_iterative_rounds=3)
    with pytest.raises(SystemExit):
        ev._decoupled_preflight(args, _problems(1))


def test_generation_error_becomes_error_record(monkeypatch):
    """A generation failure must still yield an error record (never a silent 0 scored)."""
    monkeypatch.setattr(ev, "_generate_one", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(ev, "_score", lambda *a, **k: {"reward": 1.0, "score": 1.0})
    args = _args()
    gen_q, result_q = queue.Queue(), queue.Queue()
    scorer = threading.Thread(target=ev._decoupled_score_worker, args=(args, gen_q, result_q), daemon=True)
    scorer.start()
    ev._decoupled_gen_worker(args, _problems(1)[0], 0, 0, 42, gen_q, [1], threading.Lock())
    rec = result_q.get(timeout=5)
    assert rec["error"] is not None and "boom" in rec["error"]
    assert rec["metrics"]["reward"] == 0.0  # placeholder, recorded as error

#!/usr/bin/env python3
"""
validate_pyproblem.py -- deterministic quality gate for a "program-mode" (Format B/E)
problem: the solution is an ALGORITHM/HEURISTIC (a function inside a frozen scaffold),
scored by a deterministic evaluator that runs it over a fixed instance distribution.
This is the FunSearch / AlphaEvolve / OpenEvolve shape.

Problem directory:
    statement.md         task, the function to implement, objective, instance distribution
    evaluator.py         FROZEN scaffold + scorer. CLI: `python3 evaluator.py <solution.py>`
                         -> imports the evolved function, runs the scaffold over its baked-in
                            seeded instances, prints:
                              Ratio: <mean normalized score in [0,1]>
                              Vector: [r1, r2, ...]        # per-instance normalized scores
                         A crashing / wrong-signature / cheating solution must score ~0.
    config.yaml          {checker: evaluator.py, n_instances, memory, time}
    solutions/{trivial,greedy,strong,invalid}.py   each defines the evolved function;
                         first line `# TIER: <name>`. invalid = crashes / bad output -> 0.

Gates (deterministic, no wall-time in the score -- timeouts are safety only):
    G1 import        evaluator.py + every solution import without error
    G3 bounds        every Ratio in [0,1]
    G4 determinism   evaluator gives the same Ratio on a repeated run
    G5 feasibility   the `invalid` solution scores ~0  -> evaluator sandboxes/validates
    G6 baseline      `trivial` scores a calibrated, non-perfect baseline
    G7 discrimination strong - trivial >= --margin
    G8 divergence    execution-grounded divergence across the ladder's Vectors >= --div
Exit 0 == PASS; writes <probdir>/validation.json.
"""
import argparse, json, math, re, shutil, subprocess, sys
from pathlib import Path

RATIO_RE = re.compile(r"Ratio:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
VECTOR_RE = re.compile(r"Vector:\s*(\[[^\]]*\])")
TIER_RE = re.compile(r"(?://|#)\s*TIER:\s*(\w+)", re.IGNORECASE)

import os
HARNESS_DIR = str(Path(__file__).resolve().parent)  # so evaluators can `import isorun`


def run_capture(cmd, timeout, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = HARNESS_DIR + os.pathsep + env.get("PYTHONPATH", "")
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, cwd=cwd, env=env)
        return (r.returncode, r.stdout.decode("utf-8", "replace"),
                r.stderr.decode("utf-8", "replace"), False)
    except subprocess.TimeoutExpired:
        return None, "", "", True


def parse_config(pdir):
    cfg = {"checker": "evaluator.py", "time_s": 60.0, "n_instances": None}
    txt = (pdir / "config.yaml").read_text()
    m = re.search(r"checker:\s*(\S+)", txt)
    if m: cfg["checker"] = m.group(1)
    m = re.search(r"time:\s*(\d+)\s*s", txt)
    if m: cfg["time_s"] = float(m.group(1))
    m = re.search(r"n_instances:\s*(\d+)", txt)
    if m: cfg["n_instances"] = int(m.group(1))
    return cfg


def tier_of(src):
    try:
        return (TIER_RE.search(src.read_text(errors="replace").splitlines()[0]) or [None, "unknown"])[1].lower()
    except Exception:
        return "unknown"


def _finite01(v):
    return isinstance(v, float) and v == v and v not in (float("inf"), float("-inf")) and -1e-9 <= v <= 1.0 + 1e-6


def score_solution(evaluator, sol, timeout):
    rc, out, err, to = run_capture(["python3", str(evaluator), str(sol)], timeout=timeout)
    blob = (out or "") + "\n" + (err or "")
    if to:
        return 0.0, [], "TLE"
    if rc is None or rc != 0:               # candidate crashed the evaluator / nonzero exit -> reject
        return 0.0, [], f"EVAL_RC({rc})"
    ms = RATIO_RE.findall(blob)             # take the LAST Ratio (candidate can't inject an earlier one)
    if not ms:
        return 0.0, [], "NO_RATIO"
    try:
        ratio = float(ms[-1])
    except ValueError:
        return 0.0, [], "BAD_RATIO"
    if not _finite01(ratio):
        return 0.0, [], "RATIO_OOB"
    vec = []
    vms = VECTOR_RE.findall(blob)
    if vms:
        try:
            vec = [float(x) for x in json.loads(vms[-1])]
        except Exception:
            vec = []
    return max(0.0, min(1.0, ratio)), vec, "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probdir")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--div", type=float, default=0.03)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--triv-lo", type=float, default=0.03, help="min trivial ratio (G6)")
    ap.add_argument("--triv-hi", type=float, default=0.35, help="max trivial ratio (G6)")
    args = ap.parse_args()

    pdir = Path(args.probdir).resolve()
    rep = {"probdir": str(pdir), "mode": "program", "gates": {}, "metrics": {},
           "solutions": {}, "verdict": "FAIL", "errors": []}

    def fail(g, m): rep["gates"][g] = False; rep["errors"].append(f"{g}: {m}")
    def ok(g): rep["gates"].setdefault(g, True)

    stmt_ok = (pdir / "statement.md").exists() or (pdir / "statement.txt").exists()
    if not stmt_ok: fail("G0_files", "missing statement.md")
    if not (pdir / "config.yaml").exists(): fail("G0_files", "missing config.yaml")
    cfg = parse_config(pdir)
    evaluator = pdir / cfg["checker"]
    if not evaluator.exists(): fail("G0_files", f"missing evaluator {cfg['checker']}")
    sol_dir = pdir / "solutions"
    sols = sorted(sol_dir.glob("*.py")) if sol_dir.exists() else []
    if not sols: fail("G0_files", "no solutions/*.py")
    if rep["gates"].get("G0_files") is False:
        _finish(rep, pdir); return 1
    ok("G0_files")

    # ---- G1 import checks ----
    rc, out, err, to = run_capture(["python3", "-c", f"import ast;ast.parse(open('{evaluator}').read())"], 30)
    if rc != 0: fail("G1_import", f"evaluator.py:\n{err[:800]}")
    for s in sols:
        rc, out, err, to = run_capture(["python3", "-c", f"import ast;ast.parse(open('{s}').read())"], 30)
        if rc != 0: fail("G1_import", f"{s.name}:\n{err[:400]}")
    if rep["gates"].get("G1_import") is False:
        _finish(rep, pdir); return 1
    ok("G1_import")

    # ---- score every solution ----
    vecs = {}
    bounds_ok = True
    for s in sols:
        label = s.stem; tier = tier_of(s)
        r, vec, st = score_solution(evaluator, s, args.timeout)
        if r < -1e-9 or r > 1.0 + 1e-6: bounds_ok = False
        vecs[label] = (r, vec, tier)
        rep["solutions"][label] = {"tier": tier, "ratio": round(r, 6),
                                   "vector": [round(x, 4) for x in vec], "status": st}
    ok("G3_bounds") if bounds_ok else fail("G3_bounds", "ratio outside [0,1]")

    # ---- G3b vector integrity: elements in [0,1], length == n_instances, ratio a plausible
    #      aggregate (within [min,max] of the vector -> rejects an INJECTED ratio, allows mean/gmean) ----
    vi_bad = []
    for label, (r, vec, tier) in vecs.items():
        if not vec:
            vi_bad.append(f"{label}:no-vector"); continue
        if any((x != x) or x < -1e-9 or x > 1.0 + 1e-6 for x in vec):
            vi_bad.append(f"{label}:elem-oob")
        if cfg["n_instances"] and len(vec) != cfg["n_instances"]:
            vi_bad.append(f"{label}:len{len(vec)}!={cfg['n_instances']}")
        if not (min(vec) - 1e-6 <= r <= max(vec) + 1e-6):
            vi_bad.append(f"{label}:ratio{round(r,4)}_outside[{round(min(vec),3)},{round(max(vec),3)}]")
    ok("G3b_vector") if not vi_bad else fail("G3b_vector", "; ".join(vi_bad))

    # ---- G4 determinism: re-run EVERY solution; ratio + full vector must reproduce ----
    det_ok = True; det_msg = ""
    for s in sols:
        label = s.stem
        r2, v2, _ = score_solution(evaluator, s, args.timeout)
        r1, v1, _ = vecs[label][0], vecs[label][1], None
        if abs(r2 - r1) > 1e-6 or [round(x, 6) for x in v2] != [round(x, 6) for x in v1]:
            det_ok = False; det_msg = f"{label} nondeterministic (r {r1}->{r2})"; break
    ok("G4_determinism") if det_ok else fail("G4_determinism", det_msg)

    # ---- G5b adversarial: malformed stdin->stdout candidates the evaluator must score ~0 ----
    import tempfile
    adv_src = {
        "empty":   "import sys\n",                                              # prints nothing
        "garbage": "print('not json at all')\n",                                # unparseable
        "null":    "import json\nprint(json.dumps(None))\n",                    # null answer
        "raises":  "import sys,json\njson.load(sys.stdin)\nraise SystemExit(3)\n",  # nonzero exit
        "nan":     "import sys,json\njson.load(sys.stdin)\nprint('[NaN, Infinity, -Infinity]')\n",  # non-finite
        "fakeratio": "import sys,json\njson.load(sys.stdin)\nprint('Ratio: 1.0'); print(json.dumps(None))\n",  # inject
    }
    adv_bad = []
    with tempfile.TemporaryDirectory() as td:
        for k, src in adv_src.items():
            p = Path(td) / f"adv_{k}.py"; p.write_text(src)
            r, _, _ = score_solution(evaluator, p, args.timeout)
            if r > 0.02:
                adv_bad.append((k, round(r, 3)))
    ok("G5b_adversarial") if not adv_bad else fail("G5b_adversarial", f"bad candidates scored >0.02: {adv_bad}")

    # ---- G5c ISOLATION: candidates must run OS-sandboxed (bwrap via isorun), unable to reach the
    #      parent judge (/proc), the co-located ground-truth source (synth tree), or evaluator frames.
    # (1) evaluator must actually route candidates through isorun.run_candidate
    ev_src = evaluator.read_text()
    uses_isorun = ("isorun" in ev_src and "run_candidate" in ev_src)
    # (2) frame-walk probe surfaces ONLY if the evaluator runs the candidate in-process (stderr shared)
    iso_src = (
        "import sys, json\n"
        "try: inst = json.load(sys.stdin)\n"
        "except Exception: inst = None\n"
        "seen = []\n"
        "try:\n"
        "    f = sys._getframe().f_back\n"
        "    while f is not None:\n"
        "        for nm, v in list(f.f_locals.items()):\n"
        "            if callable(v) and getattr(v,'__name__','') in ('score','grads','make_instances','baseline','run_candidate','main','build_instance'):\n"
        "                seen.append(nm)\n"
        "        f = f.f_back\n"
        "except Exception: pass\n"
        "sys.stderr.write('ISO_BREACH:'+','.join(seen))\n"
        "print(json.dumps(None))\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "adv_iso.py"; p.write_text(iso_src)
        rc, out, err, to = run_capture(["python3", str(evaluator), str(p)], timeout=args.timeout, cwd=str(pdir))
        breach = re.search(r"ISO_BREACH:(\S+)", err or "")
    # (3) probe isorun DIRECTLY: a sandboxed candidate must see a tiny /proc and NOT read the synth tree
    iso_ok, iso_msg = True, ""
    try:
        import importlib.util as _u
        _s = _u.spec_from_file_location("isorun", str(HARNESS_DIR + "/isorun.py"))
        _iso = _u.module_from_spec(_s); _s.loader.exec_module(_iso)
        probe = ("import sys,json,os\n"
                 "json.load(sys.stdin)\n"
                 "r={}\n"
                 "try: r['nproc']=len([d for d in os.listdir('/proc') if d.isdigit()])\n"
                 "except Exception: r['nproc']=-1\n"
                 "try:\n"
                 f"    open({json.dumps(str(HARNESS_DIR)+'/isorun.py')}).read(5); r['synth']='READ'\n"
                 "except Exception: r['synth']='BLOCKED'\n"
                 "print(json.dumps(r))\n")
        with tempfile.TemporaryDirectory() as td:
            pp = Path(td) / "probe.py"; pp.write_text(probe)
            ans, st = _iso.run_candidate(str(pp), {"x": 1}, timeout=args.timeout)
        if not _iso.sandbox_available():
            iso_ok, iso_msg = False, "bwrap not available -> candidates NOT OS-sandboxed"
        elif st != "OK" or not isinstance(ans, dict):
            iso_ok, iso_msg = False, f"isorun probe failed ({st})"
        elif ans.get("synth") == "READ":
            iso_ok, iso_msg = False, "sandbox candidate could READ the synth/ground-truth tree"
        elif ans.get("nproc", 999) > 20:
            iso_ok, iso_msg = False, f"sandbox candidate saw {ans.get('nproc')} procs (parent /proc not hidden)"
    except Exception as e:
        iso_ok, iso_msg = False, f"isorun probe error: {e}"
    if breach and breach.group(1).strip():
        fail("G5c_isolation", f"candidate saw evaluator internals via frame-walk: {breach.group(1)[:80]} (run via isorun)")
    elif not uses_isorun:
        fail("G5c_isolation", "evaluator does not route candidates through isorun.run_candidate")
    elif not iso_ok:
        fail("G5c_isolation", iso_msg)
    else:
        ok("G5c_isolation")

    def best_tier(t):
        v = [vecs[l][0] for l in vecs if vecs[l][2] == t]
        return max(v) if v else None

    triv = best_tier("trivial") if any(vecs[l][2] == "trivial" for l in vecs) else best_tier("baseline")
    strong = best_tier("strong")
    greedy = best_tier("greedy")
    inv = [vecs[l][0] for l in vecs if vecs[l][2] == "invalid"]

    # G5 feasibility
    if inv:
        mi = max(inv); rep["metrics"]["invalid_mean"] = round(mi, 6)
        ok("G5_feasibility") if mi <= 0.01 else fail("G5_feasibility", f"invalid scored {mi:.4f}")
    else:
        rep["gates"]["G5_feasibility"] = None

    # G6 baseline (tightened window)
    rep["metrics"]["trivial_mean"] = None if triv is None else round(triv, 6)
    if triv is None: rep["gates"]["G6_baseline"] = None
    elif triv > args.triv_hi: fail("G6_baseline", f"trivial {triv:.4f} > {args.triv_hi}")
    elif triv < args.triv_lo: fail("G6_baseline", f"trivial {triv:.4f} < {args.triv_lo}")
    else: ok("G6_baseline")

    # G7 discrimination
    rep["metrics"]["strong_mean"] = None if strong is None else round(strong, 6)
    rep["metrics"]["greedy_mean"] = None if greedy is None else round(greedy, 6)
    base = triv if triv is not None else greedy
    if strong is None or base is None:
        fail("G7_discrimination", "need strong + trivial/greedy")
    else:
        gap = strong - base; rep["metrics"]["strong_minus_trivial"] = round(gap, 6)
        ok("G7_discrimination") if gap >= args.margin else fail("G7_discrimination", f"gap {gap:.4f} < {args.margin}")

    # G8 divergence from per-instance vectors
    keys = [l for l in vecs if vecs[l][2] != "invalid" and vecs[l][1]]
    if len(keys) >= 2 and all(len(vecs[k][1]) == len(vecs[keys[0]][1]) for k in keys):
        m = len(vecs[keys[0]][1]); pairs = 0; acc = 0.0
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                va, vb = vecs[keys[a]][1], vecs[keys[b]][1]
                acc += math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb))) / math.sqrt(m); pairs += 1
        div = acc / pairs if pairs else 0.0
        rep["metrics"]["exec_divergence"] = round(div, 6)
        ok("G8_divergence") if div >= args.div else fail("G8_divergence", f"div {div:.4f} < {args.div}")
    else:
        fail("G8_divergence", "need >=2 non-invalid solutions emitting equal-length Vector:")

    hard = [v for v in rep["gates"].values() if v is not None]
    rep["verdict"] = "PASS" if all(hard) else "FAIL"
    return _finish(rep, pdir)


def _finish(rep, pdir):
    (pdir / "validation.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"verdict": rep["verdict"], "gates": rep["gates"],
                      "metrics": {k: rep["metrics"].get(k) for k in
                                  ("trivial_mean", "greedy_mean", "strong_mean",
                                   "strong_minus_trivial", "exec_divergence", "invalid_mean")},
                      "errors": rep["errors"]}, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

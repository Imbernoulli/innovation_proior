#!/usr/bin/env python3
"""
validate_problem.py -- deterministic quality gate for a synthetic FrontierSmith-style
open-ended optimization problem.

A problem directory must contain:
    statement.txt      markdown problem statement
    gen.cpp            testlib generator; `./gen <testId>` prints one test to stdout
    chk.cc             testlib checker; invoked `chk in out ans`, must print `Ratio: <float in [0,1]>`
    config.yaml        {checker, memory, time, subtasks:[{n_cases, score}], type}
    solutions/         reference solutions, each a .cpp or .py whose FIRST line is a
                       comment `// TIER: <trivial|greedy|strong|invalid>` (or `# TIER:`)

This harness replaces FrontierSmith's LLM-only cross-validation with an EXECUTION-GROUNDED
gate. It confirms, mechanically:
  G1 compile        gen.cpp, chk.cc and every solution compile
  G2 generate       every test case is produced, non-empty, within size/time budget
  G3 bounds         every parsed ratio lies in [0,1]
  G4 determinism    the checker returns the same ratio on a repeated run
  G5 feasibility    an `invalid` solution (infeasible output) scores ~0  -> checker really validates
  G6 baseline       the `trivial` solution scores a calibrated, non-perfect baseline
  G7 discrimination `strong` mean ratio beats `trivial` mean ratio by >= --margin
  G8 divergence     execution-grounded idea-divergence across solutions >= --div  (per-test
                    score vectors genuinely differ -> problem admits distinct strategies)

Exit code 0 == PASS. A machine-readable report is written to <probdir>/validation.json.
"""
import argparse, json, os, re, shutil, subprocess, sys, math, tempfile
from pathlib import Path

RATIO_RE = re.compile(r"Ratio:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
RATIO_UB_RE = re.compile(r"RatioUnbounded:\s*([0-9]*\.?[0-9]+)")
TIER_RE = re.compile(r"(?://|#)\s*TIER:\s*(\w+)", re.IGNORECASE)

HARNESS_DIR = Path(__file__).resolve().parent  # contains testlib.h
SYNTH_ROOT = str(HARNESS_DIR.parent)           # hidden from sandboxed solutions
_BWRAP = shutil.which("bwrap")
_SB_ENV = {"HOME": "/tmp", "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
_BWRAP_NETNS_ERRORS = (
    "NETLINK_ROUTE",
    "CLONE_NEWNET",
    "network namespace",
    "Operation not permitted",
)


def _bwrap_netns_failed(stderr):
    return any(s in (stderr or "") for s in _BWRAP_NETNS_ERRORS)


def _solution_bwrap_cmd(cmd, exe, dst, unshare_net=True):
    inner = (cmd[:-1] + [f"/tmp/{'sol.py' if exe.endswith('.py') else 'sol'}"])
    if exe.endswith(".py"):
        inner = [sys.executable or "python3", "-I", "--", "/tmp/sol.py"]
    bcmd = [_BWRAP, "--dev-bind", "/", "/", "--tmpfs", SYNTH_ROOT, "--tmpfs", "/tmp",
            "--ro-bind", dst, f"/tmp/{'sol.py' if exe.endswith('.py') else 'sol'}",
            "--unshare-pid"]
    if unshare_net:
        bcmd.append("--unshare-net")
    bcmd += ["--unshare-ipc", "--unshare-uts", "--proc", "/proc",
             "--die-with-parent", "--new-session", "--chdir", "/tmp"]
    return bcmd + inner


def sandbox_run_solution(cmd, inf, outf, timeout, cwd):
    """Run an UNTRUSTED solution OS-sandboxed (bwrap): synth tree hidden (no reading gen/checker/
    ground-truth source), parent /proc hidden, no net. Solution reads stdin -> writes stdout.
    Falls back to a plain run if bwrap is unavailable. Returns (rc, timed_out)."""
    exe = cmd[-1]                              # binary path or script path (last token)
    tmpd = tempfile.mkdtemp(prefix="solsb_")
    try:
        dst = os.path.join(tmpd, "sol")
        shutil.copyfile(exe, dst); os.chmod(dst, 0o755)
        if _BWRAP:
            bcmd = _solution_bwrap_cmd(cmd, exe, dst)
            env = dict(_SB_ENV)
        else:
            bcmd, env = cmd, None
        with open(inf, "rb") as fi, open(outf, "wb") as fo:
            try:
                r = subprocess.run(bcmd, stdin=fi, stdout=fo, stderr=subprocess.PIPE,
                                   timeout=timeout, env=env)
                # Some managed HPC/container nodes allow bwrap mount/pid namespaces but reject
                # creating a network namespace. Keep source-tree and /proc isolation instead of
                # falling all the way back to an unsandboxed run.
                if _BWRAP and r.returncode != 0 and _bwrap_netns_failed(r.stderr.decode("utf-8", "replace")):
                    fi.seek(0)
                    fo.seek(0)
                    fo.truncate()
                    r = subprocess.run(_solution_bwrap_cmd(cmd, exe, dst, unshare_net=False),
                                       stdin=fi, stdout=fo, stderr=subprocess.PIPE,
                                       timeout=timeout, env=env)
                return r.returncode, False
            except subprocess.TimeoutExpired:
                return None, True
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def parse_ratio(blob):
    """Robust score extraction (hardened per adversarial review):
    take the LAST `Ratio:` match (so a candidate/earlier line can't inject a fake
    score before the real one), and reject non-finite / out-of-range values.
    Returns (ratio, ok). ok=False means no valid score -> caller scores 0."""
    ms = RATIO_RE.findall(blob or "")
    if not ms:
        return 0.0, False
    try:
        v = float(ms[-1])
    except ValueError:
        return 0.0, False
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0, False
    if v < -1e-9 or v > 1.0 + 1e-6:
        return 0.0, False
    return max(0.0, min(1.0, v)), True


def run(cmd, timeout, stdin=None, stdout=None, cwd=None):
    """Run a command; return (rc, stderr_text, timed_out)."""
    try:
        r = subprocess.run(cmd, timeout=timeout, stdin=stdin, stdout=stdout,
                           stderr=subprocess.PIPE, cwd=cwd)
        return r.returncode, r.stderr.decode("utf-8", "replace"), False
    except subprocess.TimeoutExpired:
        return None, "", True


def run_capture(cmd, timeout, cwd=None):
    """Run a command capturing BOTH stdout and stderr as text; return (rc, out, err, to)."""
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, cwd=cwd)
        return (r.returncode, r.stdout.decode("utf-8", "replace"),
                r.stderr.decode("utf-8", "replace"), False)
    except subprocess.TimeoutExpired:
        return None, "", "", True


def parse_config(pdir: Path):
    cfg = {"n_cases": 10, "time_s": 8.0, "memory_mb": 512, "checker": "chk.cc"}
    txt = (pdir / "config.yaml").read_text()
    m = re.search(r"n_cases:\s*(\d+)", txt)
    if m: cfg["n_cases"] = int(m.group(1))
    m = re.search(r"time:\s*(\d+)\s*s", txt)
    if m: cfg["time_s"] = float(m.group(1))
    m = re.search(r"memory:\s*(\d+)\s*m", txt)
    if m: cfg["memory_mb"] = int(m.group(1))
    m = re.search(r"checker:\s*(\S+)", txt)
    if m: cfg["checker"] = m.group(1)
    return cfg


def resolve_generator(pdir: Path, work: Path):
    """Return (gen_cmd_fn, ok, err). Supports gen.cpp (compiled) or gen.py."""
    gc, gp = pdir / "gen.cpp", pdir / "gen.py"
    if gc.exists():
        binp = work / "gen"
        ok, err = compile_cpp(gc, binp, with_testlib=True)
        return (lambda i: [str(binp), str(i)]), ok, err
    if gp.exists():
        return (lambda i: ["python3", str(gp), str(i)]), True, ""
    return None, False, "no gen.cpp or gen.py"


def resolve_checker(pdir: Path, work: Path, chk_name: str):
    """Return (chk_cmd_fn, ok, err, is_py). C++ (.cc/.cpp) compiled with testlib, or a .py checker."""
    chk = pdir / chk_name
    if chk.suffix in (".cc", ".cpp"):
        binp = work / "chk"
        ok, err = compile_cpp(chk, binp, with_testlib=True)
        return (lambda i, o, a: [str(binp), i, o, a]), ok, err, False
    if chk.suffix == ".py" and chk.exists():
        return (lambda i, o, a: ["python3", str(chk), i, o, a]), True, "", True
    return None, False, f"checker {chk_name} not found/unsupported", False


def compile_cpp(src: Path, out: Path, with_testlib: bool):
    cmd = ["g++", "-O2", "-pipe", "-std=gnu++17"]
    if with_testlib:
        cmd += ["-I", str(HARNESS_DIR)]
    cmd += ["-o", str(out), str(src)]
    rc, err, to = run(cmd, timeout=120)
    return rc == 0, err


def solution_tier(src: Path):
    try:
        first = src.read_text(errors="replace").splitlines()[0]
    except Exception:
        return "unknown"
    m = TIER_RE.search(first)
    return m.group(1).lower() if m else "unknown"


def build_runner(src: Path, workdir: Path):
    """Return (label, run_cmd, ok, err). Compiles cpp; wraps py."""
    label = src.stem
    if src.suffix == ".py":
        return label, ["python3", str(src)], True, ""
    binp = workdir / (label + ".bin")
    ok, err = compile_cpp(src, binp, with_testlib=False)
    return label, [str(binp)], ok, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probdir")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="required strong-minus-trivial mean-ratio gap (G7)")
    ap.add_argument("--div", type=float, default=0.03,
                    help="required execution-grounded divergence (G8)")
    ap.add_argument("--triv-lo", type=float, default=0.03, help="min trivial mean (G6)")
    ap.add_argument("--triv-hi", type=float, default=0.35, help="max trivial mean (G6)")
    ap.add_argument("--gen-timeout", type=float, default=40.0)
    ap.add_argument("--size-cap-mb", type=float, default=25.0)
    ap.add_argument("--time-slack", type=float, default=2.0,
                    help="seconds added to the problem time limit for solution runs")
    ap.add_argument("--keep-testdata", action="store_true",
                    help="write generated tests into <probdir>/testdata/")
    args = ap.parse_args()

    pdir = Path(args.probdir).resolve()
    rep = {"probdir": str(pdir), "gates": {}, "metrics": {}, "solutions": {}, "verdict": "FAIL", "errors": []}

    def fail(gate, msg):
        rep["gates"][gate] = False
        rep["errors"].append(f"{gate}: {msg}")

    def ok(gate):
        rep["gates"].setdefault(gate, True)

    # ---- required files ----
    stmt = "statement.txt" if (pdir / "statement.txt").exists() else "statement.md"
    for f in ["config.yaml"]:
        if not (pdir / f).exists():
            fail("G0_files", f"missing {f}")
    if not (pdir / stmt).exists():
        fail("G0_files", "missing statement.txt/statement.md")
    if not ((pdir / "gen.cpp").exists() or (pdir / "gen.py").exists()):
        fail("G0_files", "missing gen.cpp or gen.py")
    sol_dir = pdir / "solutions"
    sols = sorted([p for p in sol_dir.glob("*") if p.suffix in (".cpp", ".py")]) if sol_dir.exists() else []
    if not sols:
        fail("G0_files", "no solutions/ found")
    if rep["gates"].get("G0_files") is False:
        _finish(rep, pdir); return 1
    ok("G0_files")

    cfg = parse_config(pdir)
    rep["metrics"]["config"] = cfg
    n = cfg["n_cases"]
    work = pdir / ".work"
    if work.exists(): shutil.rmtree(work)
    work.mkdir()

    # ---- G1 compile/resolve gen + checker (C++ testlib or Python) ----
    gen_cmd, g_ok, g_err = resolve_generator(pdir, work)
    chk_cmd, c_ok, c_err, chk_is_py = resolve_checker(pdir, work, cfg["checker"])
    if not g_ok: fail("G1_compile", "generator:\n" + g_err[:1500])
    if not c_ok: fail("G1_compile", "checker:\n" + c_err[:1500])

    runners = []
    for s in sols:
        label, cmd, s_ok, s_err = build_runner(s, work)
        tier = solution_tier(s)
        rep["solutions"][label] = {"tier": tier, "compiled": s_ok}
        if not s_ok:
            fail("G1_compile", f"{s.name}:\n{s_err[:800]}")
        else:
            runners.append((label, cmd, tier))
    if rep["gates"].get("G1_compile") is False:
        _finish(rep, pdir, work); return 1
    ok("G1_compile")

    # ---- G2 generate testdata ----
    ins = []
    for i in range(1, n + 1):
        inf = work / f"{i}.in"
        with open(inf, "wb") as fo:
            rc, err, to = run(gen_cmd(i), timeout=args.gen_timeout, stdout=fo, cwd=work)
        if to: fail("G2_generate", f"gen {i} timed out (> {args.gen_timeout}s)")
        elif rc != 0: fail("G2_generate", f"gen {i} rc={rc}: {err[:300]}")
        elif inf.stat().st_size == 0: fail("G2_generate", f"gen {i} produced empty input")
        elif inf.stat().st_size > args.size_cap_mb * 1e6:
            fail("G2_generate", f"gen {i} = {inf.stat().st_size/1e6:.1f}MB > cap {args.size_cap_mb}MB")
        (work / f"{i}.ans").write_text("")  # scorer problems ignore ans
        ins.append(inf)
    if rep["gates"].get("G2_generate") is False:
        _finish(rep, pdir, work); return 1
    ok("G2_generate")
    rep["metrics"]["testdata_bytes"] = [p.stat().st_size for p in ins]

    # ---- run each solution on each case, score via checker ----
    def score_checker(inf, outf):
        """Run the checker on a given (in,out) pair -> (ratio, status)."""
        crc, cout, cerr, cto = run_capture(
            chk_cmd(str(inf), str(outf), str(work / (inf.stem + ".ans"))), timeout=120, cwd=work)
        if cto:
            return 0.0, "CHK_TLE"
        if crc is not None and crc < 0:          # checker killed by a signal (segfault etc.)
            return 0.0, f"CHK_SIG({crc})"
        # a Python checker MUST exit 0 to have its score trusted (no print-then-crash);
        # testlib C++ checkers exit 7 (points) / nonzero on WA-without-Ratio, handled by parse.
        if chk_is_py and crc not in (0, None):
            return 0.0, f"CHK_RC({crc})"
        blob = (cerr or "") + "\n" + (cout or "")
        r, ok2 = parse_ratio(blob)
        return (r, "OK") if ok2 else (0.0, "NO_RATIO")

    def score_case(cmd, inf, outf):
        # untrusted solution runs OS-sandboxed (can't read the co-located ground-truth source)
        rc, to = sandbox_run_solution(cmd, inf, outf, cfg["time_s"] + args.time_slack, work)
        if to or rc != 0:
            return 0.0, ("TLE" if to else f"RE({rc})")
        return score_checker(inf, outf)

    score_vecs = {}   # label -> [ratio per case]
    bounds_ok = True
    for label, cmd, tier in runners:
        vec = []
        statuses = []
        for i in range(1, n + 1):
            outf = work / f"{i}.{label}.out"
            r, st = score_case(cmd, work / f"{i}.in", outf)
            if r < -1e-9 or r > 1.0 + 1e-6:
                bounds_ok = False
            vec.append(r)
            statuses.append(st)
        score_vecs[label] = vec
        rep["solutions"][label].update({
            "mean": round(sum(vec) / len(vec), 6),
            "scores": [round(x, 4) for x in vec],
            "statuses": statuses,
        })
    if bounds_ok: ok("G3_bounds")
    else: fail("G3_bounds", "a ratio fell outside [0,1]")

    # ---- G4 determinism: re-check EVERY (solution,case) via the checker + gen reproducibility ----
    det_ok = True; det_msg = ""
    for label, cmd, tier in runners:
        for i in range(1, n + 1):
            outf = work / f"{i}.{label}.out"
            if not outf.exists():
                continue
            r2, _ = score_checker(work / f"{i}.in", outf)
            if abs(r2 - score_vecs[label][i - 1]) > 1e-6:
                det_ok = False; det_msg = f"checker nondet {label} case {i}: {score_vecs[label][i-1]} vs {r2}"; break
        if not det_ok:
            break
    if det_ok:  # generator reproducibility (case 1)
        tmp = work / "1.regen"
        with open(tmp, "wb") as fo:
            run(gen_cmd(1), timeout=args.gen_timeout, stdout=fo, cwd=work)
        if tmp.read_bytes() != (work / "1.in").read_bytes():
            det_ok = False; det_msg = "gen(1) not reproducible across runs"
    ok("G4_determinism") if det_ok else fail("G4_determinism", det_msg)

    # ---- G5b adversarial feasibility: malformed outputs must score ~0 ----
    adv = {"empty": "", "garbage": "qwerty\nfoo bar baz\n",
           "huge": "99999999999999999 99999999999999999 99999999999999999\n"}
    # nan/inf flood derived from a real solution's output (same shape/count) -> catches
    # checkers that don't reject non-finite values (the fsx_A_0114 class).
    triv_lbl = next((l for l in score_vecs if rep["solutions"][l]["tier"] in ("trivial", "greedy", "strong")), None)
    if triv_lbl and (work / f"1.{triv_lbl}.out").exists():
        base_out = (work / f"1.{triv_lbl}.out").read_text()
        def _sub(tok, rep_tok):
            try:
                float(tok); return rep_tok
            except ValueError:
                return tok
        def _flood(rep_tok):
            return "\n".join(" ".join(_sub(t, rep_tok) for t in ln.split()) for ln in base_out.splitlines())
        adv["nan"] = _flood("nan")
        adv["inf"] = _flood("inf")
    adv_bad = []
    for k, txt in adv.items():
        af = work / f"adv_{k}.out"; af.write_text(txt)
        r, _ = score_checker(work / "1.in", af)
        if r > 0.02:
            adv_bad.append((k, round(r, 3)))
    ok("G5b_adversarial") if not adv_bad else fail("G5b_adversarial", f"malformed outputs scored >0.02: {adv_bad}")

    # ---- collect tiers ----
    by_tier = {}
    for label in rep["solutions"]:
        t = rep["solutions"][label]["tier"]
        if label in score_vecs:
            by_tier.setdefault(t, []).append(label)

    def tier_mean(t):
        labs = by_tier.get(t, [])
        if not labs: return None
        return max(sum(score_vecs[l]) / len(score_vecs[l]) for l in labs)  # best in tier

    triv = tier_mean("trivial") if "trivial" in by_tier else tier_mean("baseline")
    strong = tier_mean("strong")
    greedy = tier_mean("greedy")
    invalid_labels = by_tier.get("invalid", [])

    # ---- G5 feasibility: invalid solution must score ~0 ----
    if invalid_labels:
        inv = max(sum(score_vecs[l]) / len(score_vecs[l]) for l in invalid_labels)
        rep["metrics"]["invalid_mean"] = round(inv, 6)
        if inv <= 0.01: ok("G5_feasibility")
        else: fail("G5_feasibility", f"invalid solution scored {inv:.4f} > 0.01 (checker not validating)")
    else:
        rep["gates"]["G5_feasibility"] = None  # not tested

    # ---- G6 baseline calibration (tightened per review: enforce trivial ~= 0.1 convention) ----
    rep["metrics"]["trivial_mean"] = None if triv is None else round(triv, 6)
    if triv is None:
        rep["gates"]["G6_baseline"] = None
    elif triv > args.triv_hi:
        fail("G6_baseline", f"trivial scores {triv:.4f} > {args.triv_hi}: baseline too generous / not the trivial reference")
    elif triv < args.triv_lo:
        fail("G6_baseline", f"trivial scores {triv:.4f} < {args.triv_lo}: baseline uncalibrated / trivial infeasible")
    else:
        ok("G6_baseline")

    # ---- G7 discrimination ----
    rep["metrics"]["strong_mean"] = None if strong is None else round(strong, 6)
    rep["metrics"]["greedy_mean"] = None if greedy is None else round(greedy, 6)
    base_for_gap = triv if triv is not None else (greedy if greedy is not None else None)
    if strong is None or base_for_gap is None:
        fail("G7_discrimination", "need both a 'strong' and a 'trivial'/'greedy' solution")
    else:
        gap = strong - base_for_gap
        rep["metrics"]["strong_minus_trivial"] = round(gap, 6)
        if gap >= args.margin: ok("G7_discrimination")
        else: fail("G7_discrimination", f"strong beats baseline by only {gap:.4f} < margin {args.margin}")

    # ---- G8 execution-grounded idea divergence ----
    #   d = mean over solution pairs of (1/sqrt(m)) * L2(scorevec_i, scorevec_j)
    keys = [l for l in score_vecs if rep["solutions"][l]["tier"] != "invalid"]
    if len(keys) >= 2:
        pairs = 0
        acc = 0.0
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                va, vb = score_vecs[keys[a]], score_vecs[keys[b]]
                l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))
                acc += l2 / math.sqrt(len(va))
                pairs += 1
        div = acc / pairs if pairs else 0.0
        rep["metrics"]["exec_divergence"] = round(div, 6)
        if div >= args.div: ok("G8_divergence")
        else: fail("G8_divergence", f"exec divergence {div:.4f} < {args.div}: solutions behave identically")
    else:
        fail("G8_divergence", "need >=2 non-invalid solutions")

    # ---- verdict: all non-None gates must be True ----
    hard = [v for v in rep["gates"].values() if v is not None]
    rep["verdict"] = "PASS" if all(hard) else "FAIL"
    rc = _finish(rep, pdir, work if not args.keep_testdata else None)
    if args.keep_testdata:
        td = pdir / "testdata"; td.mkdir(exist_ok=True)
        for i in range(1, n + 1):
            shutil.copy(work / f"{i}.in", td / f"{i}.in")
            (td / f"{i}.ans").write_text("")
        shutil.rmtree(work, ignore_errors=True)
    return rc


def _finish(rep, pdir, work=None):
    (pdir / "validation.json").write_text(json.dumps(rep, indent=2))
    if work and work.exists():
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"verdict": rep["verdict"], "gates": rep["gates"],
                      "metrics": {k: rep["metrics"].get(k) for k in
                                  ("trivial_mean", "greedy_mean", "strong_mean",
                                   "strong_minus_trivial", "exec_divergence", "invalid_mean")},
                      "errors": rep["errors"]}, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

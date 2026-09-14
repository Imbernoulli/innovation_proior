"""v2: per-run (era x arm x task) artifacts + metrics for the quantitative MLS case study.

Adds over v1: agent-dir lookup by workspace name, editable-file name resolution (agents pass basenames /
absolute paths), replication of mlsbench's leaderboard row selection so we know WHICH test's code was scored
(and whether the scored row even came from this run), and an own-run score restricted to this run's rows.
"""
import json, glob, re, os, sys, ast, difflib, tokenize, io, collections, math, textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi"
os.environ.setdefault("MLSBENCH_ROOT", f"{D}/mlsroot"); os.environ.setdefault("MLSBENCH_DATA_ROOT", f"{D}/mlsvendor/data")
sys.path.insert(0, f"{D}/mlsroot/src"); os.chdir(f"{D}/mlsroot")
from mlsbench.scoring.evaluate import BaselineAnchors, load_expanded_spec, score_record_details, _load_leaderboard_records
from mlsbench.agent.leaderboard import Leaderboard
OUT = os.path.dirname(os.path.abspath(__file__))
ARMS9 = ["base9b_v2c","ft01mix_a10","ft03nm_a20","lo32nm_a10","rlv5_base_s20","rlv5_ft01mix_a10_s20","rlv5_ft03nm_a20_s20","rlv5_lo32nm_a10_s20"]
ARMS4 = ["base4b","4b_ft01mix_a10","rlv5_4b_base_s20","rlv5_4b_ft01mix_a10_s20"]
ARMS = ARMS9 + ARMS4
TASKS = "causal-discovery-discrete causal-observational-linear-gaussian causal-observational-linear-non-gaussian causal-observational-nonlinear causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy optimization-hyperparameter-search optimization-multi-objective optimization-nas optimization-online-bandit".split()

def san(fn): return fn.replace("/", "_")
def pts(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))

# ---------------- task static info ----------------
def task_info(task):
    cfg = json.load(open(f"{D}/mlsroot/tasks/{task}/config.json"))
    fs = [f for f in cfg["files"] if "edit" in f]; assert len(fs) == 1, (task, len(fs))
    f = fs[0]; ed = f["edit"][0]; assert len(f["edit"]) == 1
    baselines = {}
    for name, b in cfg["baselines"].items():
        p = f"{D}/mlsroot/tasks/{task}/{b['edit_ops']}"
        tree = ast.parse(open(p).read())
        assert all(type(n).__name__ in ("Expr", "Assign", "ImportFrom") for n in tree.body), p
        ns = {}; exec(compile(tree, p, "exec"), {}, ns)
        ops = [o for o in ns["OPS"] if o["file"] == f["filename"] and o["op"] == "replace"]
        if not ops: continue
        assert len(ops) == 1, (task, name, len(ops))
        baselines[name] = dict(start_line=ops[0]["start_line"], end_line=ops[0]["end_line"], content=ops[0]["content"])
    return {"filename": f["filename"], "edit_start": ed["start"], "edit_end": ed["end"], "baseline_ops": baselines}

def baseline_regions(ti, tpl):
    if tpl is None: return {}
    L = tpl.splitlines(); a, b = ti["edit_start"], ti["edit_end"]; out = {}
    for name, op in ti["baseline_ops"].items():
        s0, e0 = op["start_line"], op["end_line"]; assert a <= s0 <= e0 <= b, (name, a, b, s0, e0)
        out[name] = "\n".join(L[a - 1:s0 - 1] + op["content"].splitlines() + L[e0:b]) + "\n"
    return out

# ---------------- leaderboard selection (replica of evaluate.py) ----------------
_spec_cache = {}
def task_spec(task):
    if task not in _spec_cache:
        td = Path(f"{D}/mlsroot/tasks/{task}"); anchors = BaselineAnchors(td); spec = load_expanded_spec(td, anchors)
        recs = _load_leaderboard_records(td / "leaderboard.csv")
        _spec_cache[task] = (spec, anchors, recs)
    return _spec_cache[task]

def completeness(rec):
    return sum(1 for k, v in rec.items() if not (k in {"timestamp","model","is_final","seed"} or k.startswith("elapsed_") or k.endswith("_std"))
               and v not in ("", None) and not (isinstance(v, float) and math.isnan(v)) and not (isinstance(v, str) and v.strip().lower() in {"nan","null","none"}))

def select_record(task, recs):
    """returns (record, score, valid) exactly like evaluate.py for one model's rows"""
    spec, anchors, _ = task_spec(task)
    if not recs: return None, None, None
    def valid(r): return Leaderboard.has_real_metrics(r) and score_record_details(spec, r, anchors)[2]
    def pick(rows): return max(rows, key=lambda r: (completeness(r), str(r.get("timestamp", ""))))
    fm = [r for r in recs if r.get("seed") == "mean" and str(r.get("is_final", "")).lower() == "true"]
    fa = [r for r in recs if str(r.get("is_final", "")).lower() == "true"]
    nm = [r for r in recs if r.get("seed") == "mean"]
    record = None; fallback = None
    for tier in (fm, fa, nm, recs):
        if not tier: continue
        v = [r for r in tier if valid(r)]
        if v: record = pick(v); break
        if fallback is None: fallback = pick(tier)
    if record is None: record = fallback
    if record is None: return None, None, None
    if not Leaderboard.has_real_metrics(record): return record, 0.0, False
    sc, _, rv = score_record_details(spec, record, anchors)
    return record, sc, rv

def metric_sig(rec):
    return tuple(sorted((k, v) for k, v in rec.items() if not (k in {"timestamp","model","is_final","seed"} or k.startswith("elapsed_") or k.endswith("_std")) and v not in ("", None) and not (isinstance(v, float) and math.isnan(v))))

# ---------------- locate runs ----------------
def outdir(era, arm, task):
    if era == "old": return f"{D}/outputs/cc_mls21_{arm}"
    if task == "ml-active-learning" and os.path.exists(f"{D}/outputs/cc_mls21_{arm}_p1-alfix/summary.json"):
        return f"{D}/outputs/cc_mls21_{arm}_p1-alfix"
    return f"{D}/outputs/cc_mls21_{arm}_p1"
_summ = {}
def summary(od):
    if od not in _summ: _summ[od] = json.load(open(f"{od}/summary.json"))
    return _summ[od]

def locate(era, arm, task):
    od = outdir(era, arm, task); s = summary(od)
    t = [x for x in s["tasks"] if x["task"] == task]; assert len(t) == 1; t = t[0]
    log = open(t["log"], errors="replace").read()
    m = re.search(rf"workspace/{re.escape(task)}/([^/\s]+)/", log); ws = m.group(1) if m else None
    tag = arm if era == "old" else arm + "_p1"
    cands = sorted(glob.glob(f"{D}/mlsroot/logs/{task}/vllm/{tag}__cc-*-{task}/agent/messages.jsonl"))
    ad = None
    for c in cands:
        for l in open(c):
            if '"_meta"' in l and json.loads(l).get("exp_name") == ws: ad = os.path.dirname(c); break
        if ad: break
    if ad is None:
        c2 = [c for c in cands if f"__cc-{s['slurm_job_id']}-" in c]
        ad = os.path.dirname(c2[0]) if c2 else None
    ctx400 = "maximum context length" in log
    return dict(era=era, arm=arm, task=task, tag=tag, outdir=od, job=s["slurm_job_id"], status=t["status"], score=t.get("score"),
                n_settings=len(t.get("settings") or []), agent_returncode=t.get("agent_returncode"), elapsed_s=t.get("elapsed_s"), ws=ws, agent_dir=ad,
                agent_dir_by_ws=(ad is not None and ws is not None and f"__cc-{s['slurm_job_id']}-" not in (ad or "")), ctx400=ctx400)

# ---------------- replay ----------------
def is_editable(efile, fn):
    if efile is None: return True
    return efile == fn or os.path.basename(efile) == os.path.basename(fn)

def apply_str_replace(text, old, new):
    """exact-match reconstruction of the tool's str_replace; None when not exactly one exact match"""
    if text is None or old is None: return None
    if text.count(old) == 1: return text.replace(old, new if new is not None else "")
    return None

def replay(run, ti, tpl=None):
    ad = run["agent_dir"]; fn = ti["filename"]
    msgs = [json.loads(l) for l in open(f"{ad}/messages.jsonl")]
    meta = [m for m in msgs if m.get("role") == "_meta"]; exp = meta[0]["exp_name"] if meta else None
    pairs = []; pending = None
    for m in msgs:
        if m.get("role") == "assistant": pending = m
        elif m.get("role") == "tool_result" and pending is not None: pairs.append((pending, m["result"])); pending = None
    if pending is not None: pairs.append((pending, ""))
    tok = [json.loads(l) for l in open(f"{ad}/tokens.jsonl")] if os.path.exists(f"{ad}/tokens.jsonl") else []
    step_ts = {}
    for t in tok: step_ts.setdefault(t["step"], t["timestamp"])
    stack = []; test_snap = {}; test_step = {}; cur_range = None; range_hist = {}
    counters = collections.Counter(); submitted_k = None; finalized_k = None; snap_missing = []; last_test_idx = None; mutated_after_test = False
    def top(): return stack[-1] if stack else tpl
    for c, r in pairs:
        step = c["step"]; tn = c.get("tool_name"); ti_ = c.get("tool_input") or {}
        counters["steps"] += 1
        if tn in ("edit", "undo", "reset") and last_test_idx is not None and (tn != "edit" or (r.startswith("OK") and is_editable(ti_.get("filename"), fn))): mutated_after_test = True
        m = re.search(r"editable: (\d+)[–-](\d+) \| total: (\d+) lines", r)
        if m: cur_range = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        range_hist[step] = cur_range
        if tn == "edit":
            efile = ti_.get("filename")
            if not is_editable(efile, fn):
                counters["edit_other_file"] += 1
                if r.startswith("OK"): counters["edit_other_file_ok"] += 1
            else:
                counters["edit"] += 1
                if r.startswith("OK"):
                    counters["edit_ok"] += 1
                    cands = [f"{ad}/files/step_{step}_{san(fn)}"]
                    if efile: cands += [f"{ad}/files/step_{step}_{san(efile.lstrip('/'))}", f"{ad}/files/step_{step}_{os.path.basename(efile)}"]
                    sp = next((p for p in cands if os.path.exists(p)), None)
                    if sp is None:
                        snap_missing.append(step); counters["snap_missing"] += 1
                        rec = apply_str_replace(top(), ti_.get("old_str"), ti_.get("new_str")) if "[matched after" not in r else None
                        if rec is None: counters["snap_unreconstructable"] += 1
                        else: counters["snap_reconstructed"] += 1
                        stack.append(rec)
                    else: stack.append(open(sp).read())
                    if "no longer parses as Python" in r: counters["edit_ok_unparsable"] += 1
                else:
                    counters["edit_err"] += 1
                    if "exceed the editable range" in r or "outside the editable" in r: counters["edit_err_range"] += 1
                    elif "not found" in r: counters["edit_err_notfound"] += 1
                    elif "allow_create" in r: counters["edit_err_create"] += 1
        elif tn == "undo":
            if r.startswith("Undo complete"):
                n = sum(1 for line in r.splitlines() if line.startswith("Restored: ") and is_editable(line[len("Restored: "):].strip(), fn))
                counters["undo_restored_other"] += r.count("Restored: ") - n
                for _ in range(min(n, len(stack))): stack.pop()
                counters["undo_ok"] += 1
            else: counters["undo_err"] += 1
        elif tn == "reset": stack.clear(); counters["reset"] += 1
        elif tn == "test":
            mm = re.search(r"\[Test #(\d+)\]", r)
            if mm:
                k = int(mm.group(1)); counters["test_ok"] += 1
                test_snap[k] = ((stack[-1] if stack else None), cur_range, len(stack)); test_step[k] = step; last_test_idx = k; mutated_after_test = False
                if "Traceback" in r: counters["test_traceback"] += 1
            else: counters["test_err"] += 1
        elif tn == "submit":
            mm = re.search(r"Submitting result from test #(\d+)", r)
            if mm: submitted_k = int(mm.group(1)); counters["submit_ok"] += 1
            else: counters["submit_err"] += 1
        elif tn == "view": counters["view"] += 1
        else: counters["other_tool"] += 1
        mf = re.search(r"Finalized test #(\d+)", r)
        if mf: finalized_k = int(mf.group(1))
        if "No valid metrics available" in r: counters["no_valid_metrics"] += 1
    asum = json.load(open(f"{ad}/summary.json")) if os.path.exists(f"{ad}/summary.json") else {}
    return dict(exp_name=exp, final_stack_top=(stack[-1] if stack else None), n_stack=len(stack), test_snap=test_snap, test_step=test_step, step_ts=step_ts,
                last_test_idx=last_test_idx, mutated_after_last_test=mutated_after_test, stack_top_unknown=(bool(stack) and stack[-1] is None),
                cur_range=cur_range, counters=dict(counters), submitted_k=submitted_k, finalized_k=finalized_k, snap_missing=snap_missing,
                n_calls=len(tok), completion_tokens=sum(t.get("completion_tokens", 0) for t in tok), first_prompt_tokens=(tok[0]["prompt_tokens"] if tok else None),
                first_ts=(tok[0]["timestamp"] if tok else None), wrote_at=asum.get("wrote_at"), agent_done=asum.get("done"), agent_tests=asum.get("tests"))

def leaderboard_view(run, rp):
    """which row did mlsbench pick for this model, is it from this run, which test does it correspond to; own-run score"""
    task = run["task"]; model = f"vllm/{run['tag']}"
    _, _, recs = task_spec(task)
    mine = [r for r in recs if str(r.get("model")) == model]
    rec, sc, rv = select_record(task, mine)
    out = dict(lb_rows_model=len(mine), lb_selected_score=sc, lb_selected_valid=rv)
    if rec is None: return out
    if rp["first_ts"] is None: return out | dict(lb_selected_in_run=None)
    t0 = pts(rp["first_ts"]) - timedelta(seconds=5); t1 = (pts(rp["wrote_at"]) if rp["wrote_at"] else t0 + timedelta(hours=3)) + timedelta(seconds=120)
    inwin = [r for r in mine if t0 <= pts(str(r["timestamp"])) <= t1]
    out["lb_rows_in_run"] = len(inwin)
    out["lb_selected_in_run"] = t0 <= pts(str(rec["timestamp"])) <= t1
    out["lb_selected_is_final"] = str(rec.get("is_final", "")).lower() == "true"
    # own-run selection
    rec2, sc2, rv2 = select_record(task, inwin)
    out["own_score"] = sc2 if rec2 is not None else None; out["own_valid"] = rv2
    # map selected (or own) record -> test k via timestamp of a non-final row with identical metrics in window
    def k_for(record):
        if record is None: return None
        sig = metric_sig(record); ts_sorted = sorted(rp["step_ts"].items(), key=lambda kv: pts(kv[1]))
        cands = [r for r in inwin if metric_sig(r) == sig and str(r.get("is_final", "")).lower() != "true"] or ([record] if t0 <= pts(str(record["timestamp"])) <= t1 else [])
        ks = set()
        for r in cands:
            rt = pts(str(r["timestamp"])); step = None
            for s, ts in ts_sorted:
                if pts(ts) <= rt: step = s
            for k, st in rp["test_step"].items():
                if st == step: ks.add(k)
        return sorted(ks)
    out["lb_selected_tests"] = k_for(rec) if out["lb_selected_in_run"] else []
    out["own_tests"] = k_for(rec2) if rec2 is not None else []
    return out

# ---------------- metrics ----------------
class CC(ast.NodeVisitor):
    def __init__(self): self.cc = 0; self.depth = 0; self.maxdepth = 0
    def _branch(self, node, add=1):
        self.cc += add; self.depth += 1; self.maxdepth = max(self.maxdepth, self.depth); self.generic_visit(node); self.depth -= 1
    def visit_If(self, n): self._branch(n)
    def visit_For(self, n): self._branch(n)
    def visit_AsyncFor(self, n): self._branch(n)
    def visit_While(self, n): self._branch(n)
    def visit_With(self, n): self._branch(n, 0)
    def visit_Try(self, n): self._branch(n, len(n.handlers))
    def visit_IfExp(self, n): self.cc += 1; self.generic_visit(n)
    def visit_BoolOp(self, n): self.cc += len(n.values) - 1; self.generic_visit(n)
    def visit_comprehension(self, n): self.cc += 1 + len(n.ifs); self.generic_visit(n)
    def visit_Match(self, n): self._branch(n, len(n.cases))
    def visit_FunctionDef(self, n): self.cc += 1; self._branch(n, 0)
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, n): self._branch(n, 0)

def code_metrics(text):
    if text is None: return None
    src = textwrap.dedent(text); lines = src.splitlines()
    m = dict(lines=len(lines), loc=sum(1 for l in lines if l.strip() and not l.strip().startswith("#")), chars=len(src))
    try: tree = ast.parse(src); m["parse_ok"] = True
    except SyntaxError: m["parse_ok"] = False; return m
    nodes = list(ast.walk(tree)); m["ast_nodes"] = len(nodes)
    v = CC(); v.visit(tree); m["cyclomatic"] = v.cc; m["max_nesting"] = v.maxdepth
    m["n_funcs"] = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes)
    m["n_classes"] = sum(isinstance(n, ast.ClassDef) for n in nodes)
    m["n_calls"] = sum(isinstance(n, ast.Call) for n in nodes)
    m["n_num_literals"] = sum(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) for n in nodes)
    m["n_imports"] = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)
    m["n_loops"] = sum(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in nodes)
    m["n_try"] = sum(isinstance(n, ast.Try) for n in nodes)
    names = set()
    for n in nodes:
        if isinstance(n, ast.Name): names.add(n.id)
        elif isinstance(n, ast.Attribute): names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): names.add(n.name)
    m["n_identifiers"] = len(names)
    imports = set()
    for n in nodes:
        if isinstance(n, ast.Import): imports |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module: imports.add(n.module.split(".")[0])
    m["imports"] = sorted(imports)
    return m

def toks(text):
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING): out.append(t.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = re.findall(r"\w+|[^\w\s]", text)
    return out

def sim(a, b):
    if a is None or b is None: return None
    ta, tb = toks(a), toks(b); r = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
    la = set(l.strip() for l in a.splitlines() if l.strip()); lb = set(l.strip() for l in b.splitlines() if l.strip())
    ia = set(re.findall(r"[A-Za-z_]\w*", a)); ib = set(re.findall(r"[A-Za-z_]\w*", b))
    return dict(tok_ratio=r, line_jaccard=len(la & lb) / max(1, len(la | lb)), ident_jaccard=len(ia & ib) / max(1, len(ia | ib)), lines_kept_frac=len(la & lb) / max(1, len(lb)))

def region(fulltext, rng):
    if fulltext is None or rng is None: return None
    a, b, _ = rng; return "\n".join(fulltext.splitlines()[a - 1:b]) + "\n"

# ---------------- main ----------------
def main():
    tinfo = {t: task_info(t) for t in TASKS}
    runs = []; problems = []
    for era in ("old", "p1"):
        for arm in ARMS:
            for task in TASKS:
                run = locate(era, arm, task); ti = tinfo[task]
                ws_file = f"{D}/mlsroot/vendor/workspace/{task}/{run['ws']}/{ti['filename']}" if run["ws"] else None
                ws_text = open(ws_file).read() if ws_file and os.path.exists(ws_file) else None
                run["ws_exists"] = ws_text is not None; run["_ws_text"] = ws_text
                if run["agent_dir"] is None: problems.append((era, arm, task, "no agent dir")); runs.append(run); continue
                rp = replay(run, ti)
                run["_rp0"] = rp; run["final_is_template"] = (rp["n_stack"] == 0); run["counters"] = rp["counters"]
                runs.append(run)
    # templates = workspace file of runs that never had a successful edit on the editable file
    templates = {}
    for task in TASKS:
        cands = [r["_ws_text"] for r in runs if r["task"] == task and r.get("final_is_template") and r["_ws_text"]]
        uniq = collections.Counter(cands)
        templates[task] = dict(n_cands=len(cands), n_unique=len(uniq), text=(uniq.most_common(1)[0][0] if uniq else None), agreement=(uniq.most_common(1)[0][1] / len(cands) if cands else None))
    # second pass with template known (for reconstruction of missing snapshots)
    for r in runs:
        if r["agent_dir"] is None: r["no_trajectory"] = True; continue
        r["no_trajectory"] = False
        tpl = templates[r["task"]]["text"]; ti = tinfo[r["task"]]
        rp = replay(r, ti, tpl); del r["_rp0"]
        r.update({k: v for k, v in rp.items() if k not in ("final_stack_top", "test_snap", "test_step", "step_ts")})
        r["exp_matches_ws"] = (rp["exp_name"] == r["ws"])
        r.update(leaderboard_view(r, rp))
        # final text: replay top, else template, else (unknown) workspace file
        if rp["n_stack"] == 0: final = tpl; src = "template"
        elif rp["final_stack_top"] is not None: final = rp["final_stack_top"]; src = "replay"
        else: final = r["_ws_text"]; src = "workspace_fallback"
        r["_final_text"] = final; r["final_source"] = src
        r["replay_ok"] = (r["_ws_text"] == final) if (src != "workspace_fallback" and r["_ws_text"] is not None and final is not None) else None
        if r["replay_ok"] is False: problems.append((r["era"], r["arm"], r["task"], "replay mismatch", rp["n_stack"], rp["counters"]))
        # scored test: leaderboard-derived (prefer the submitted one when several tests share identical metrics), else submit marker
        ks = r.get("lb_selected_tests") if r.get("lb_selected_in_run") else (r.get("own_tests") or [])
        ks = ks or []
        if ks: k = r["submitted_k"] if r.get("submitted_k") in ks else ks[-1]; src_k = "leaderboard" if r.get("lb_selected_in_run") else "own_leaderboard"
        else: k = r.get("submitted_k") or r.get("finalized_k"); src_k = "submit" if r.get("submitted_k") else ("finalized" if r.get("finalized_k") else None)
        r["scored_k"] = k; r["scored_k_source"] = src_k
        r["never_tested"] = rp["counters"].get("test_ok", 0) == 0
        snap = rp["test_snap"].get(k) if k else None
        if snap is None:
            r["_scored_text"] = None; r["scored_range"] = None; r["scored_is_template"] = None; r["scored_source"] = None
        else:
            txt, rng, depth = snap
            if depth == 0: r["_scored_text"] = tpl; r["scored_source"] = "template"
            elif txt is not None: r["_scored_text"] = txt; r["scored_source"] = "replay"
            elif k == rp["last_test_idx"] and not rp["mutated_after_last_test"]: r["_scored_text"] = final; r["scored_source"] = "final_fallback"
            else: r["_scored_text"] = None; r["scored_source"] = "unknown"
            r["scored_range"] = rng; r["scored_is_template"] = (depth == 0)
    # metrics
    for r in runs:
        ti = tinfo[r["task"]]; tpl = templates[r["task"]]["text"]
        tpl_rng = (ti["edit_start"], ti["edit_end"], len(tpl.splitlines()) if tpl else None)
        tpl_region = region(tpl, tpl_rng) if tpl else None
        r["template_region_metrics"] = code_metrics(tpl_region); bregs = baseline_regions(ti, tpl)
        out = {}
        for which, text in (("final", r.get("_final_text")), ("scored", r.get("_scored_text"))):
            if text is None: out[which] = None; continue
            rng = r.get("cur_range") if which == "final" else r.get("scored_range")
            if text == tpl: rng = tpl_rng
            reg = region(text, rng) if rng else text
            cm = code_metrics(reg); wm = code_metrics(text)
            d = dict(region_metrics=cm, whole_parse_ok=wm["parse_ok"], region_fallback=rng is None, sim_template=sim(reg, tpl_region) if tpl_region else None,
                     sim_baselines={b: sim(reg, bt) for b, bt in bregs.items()}, identical_to_template=(text == tpl))
            if d["sim_baselines"]:
                nb = max(d["sim_baselines"], key=lambda b: d["sim_baselines"][b]["tok_ratio"]); d["nearest_baseline"] = nb; d["sim_nearest_baseline"] = d["sim_baselines"][nb]
            out[which] = d
        r["metrics"] = out
        r["scored_equals_final"] = (r.get("_scored_text") == r.get("_final_text")) if r.get("_scored_text") is not None else None
    texts = {f"{r['era']}|{r['arm']}|{r['task']}": dict(final=r.pop("_final_text", None), scored=r.pop("_scored_text", None), ws=r.pop("_ws_text", None)) for r in runs}
    json.dump(dict(runs=runs, templates={t: {k: v for k, v in d.items() if k != "text"} for t, d in templates.items()},
                   tinfo={t: {k: v for k, v in d.items() if k != "baseline_ops"} | {"baseline_names": list(d["baseline_ops"])} for t, d in tinfo.items()}), open(f"{OUT}/runs.json", "w"), indent=1)
    json.dump(texts, open(f"{OUT}/texts.json", "w")); json.dump({t: d["text"] for t, d in templates.items()}, open(f"{OUT}/templates.json", "w"))
    json.dump({t: baseline_regions(tinfo[t], templates[t]["text"]) for t in TASKS}, open(f"{OUT}/baselines.json", "w"))
    # report
    print("runs", len(runs), "problems", len(problems))
    for p in problems: print("PROBLEM", p)
    print("templates:", {t: (d["n_cands"], d["n_unique"], round(d["agreement"], 2) if d["agreement"] else None) for t, d in templates.items()})
    print("agent_dir_by_ws (job differs from summary):", collections.Counter((r["era"], r.get("agent_dir_by_ws")) for r in runs))
    print("replay_ok:", collections.Counter((r["era"], r.get("replay_ok")) for r in runs))
    print("snap_missing runs:", sum(1 for r in runs if r.get("counters", {}).get("snap_missing")))
    print("lb_selected_in_run:", collections.Counter((r["era"], r.get("lb_selected_in_run")) for r in runs))
    print("official score == lb_selected_score:", collections.Counter((r["era"], (r.get("score") is None and r.get("lb_selected_score") in (None, 0.0)) or (r.get("score") is not None and r.get("lb_selected_score") is not None and abs(r["score"] - r["lb_selected_score"]) < 1e-9)) for r in runs))
    print("own_score differs from official (>1e-6):", [(r["era"], r["arm"], r["task"], r.get("score"), r.get("own_score")) for r in runs if r.get("own_score") is not None and r.get("score") is not None and abs(r["own_score"] - r["score"]) > 1e-6][:30])
    print("scored_k_source:", collections.Counter((r["era"], r.get("scored_k_source")) for r in runs))
    print("scored_k vs submitted_k mismatch:", [(r["era"], r["arm"], r["task"], r.get("scored_k"), r.get("submitted_k"), r.get("lb_selected_tests")) for r in runs if r.get("submitted_k") and r.get("scored_k") and r["scored_k"] != r["submitted_k"]][:20])
    print("status x has scored text:", collections.Counter((r["status"], r.get("metrics", {}).get("scored") is not None) for r in runs))
    print("scored_source None by (never_tested, status, lb_in_run):", collections.Counter((r.get("never_tested"), r["status"], r.get("lb_selected_in_run")) for r in runs if r.get("scored_source") is None and not r.get("no_trajectory")))
    print("final_source:", collections.Counter((r["era"], r.get("final_source")) for r in runs))
    print("scored_source:", collections.Counter((r["era"], r.get("scored_source")) for r in runs))
    print("snap reconstruct:", sum(r.get("counters", {}).get("snap_reconstructed", 0) for r in runs), "unrecon:", sum(r.get("counters", {}).get("snap_unreconstructable", 0) for r in runs))
if __name__ == "__main__": main()

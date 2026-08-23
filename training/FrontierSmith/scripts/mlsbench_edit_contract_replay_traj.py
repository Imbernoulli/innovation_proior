#!/usr/bin/env python3
"""Replay whole banked transcripts through the real edit path under three modes.

  strict : byte-exact matcher, no syntax gate      (original production)
  landed : tolerant unique-match ladder, no gate   (the already-landed patch)
  new    : ladder + syntax gate + full-region echo (this change)

Caveat, stated up front: the model's actions are FROZEN. op='rewrite' never
appears in a banked transcript, so this replay cannot show the main win -- it
measures only the parts that act on the same frozen actions (matcher, syntax
gate). Its purpose here is a regression check: the syntax gate must not inflate
the rejection rate much, and it must raise the share of runs whose final file
still parses.
"""
import ast
import collections
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "repo"))
from mlsbench.agent.tools import WorkspaceTools  # noqa: E402

LOGS = Path("/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/logs")
FIX = Path("/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/outputs/cc_mls_editab/fixtures.pkl")
import pickle  # noqa: E402
fixtures = pickle.load(open(FIX, "rb"))


class StubTools(WorkspaceTools):
    def __init__(self, filename, path, editable, use_replace=True):
        self._path = path
        self.config_edit = [{"filename": filename,
                             "edit": [{"start": s, "end": e} for s, e in editable]}]
        self.config_task = {"allow_create": False}
        self.all_external_packages = [filename.split("/")[0]]
        self.live_protected_ranges = {
            filename: self._allowed_to_protected([[s, e] for s, e in editable])}
        self._history = []
        self._created_files = set()
        self.step_count = 0
        self.use_replace = use_replace
        self.allow_rewrite = use_replace

    def _resolve_workspace_path(self, filename):
        return self._path

    def _find_workspace_pkg(self, pkg):
        raise FileNotFoundError(pkg)


def setmode(mode):
    os.environ["MLSBENCH_STRICT_STR_REPLACE"] = "1" if mode == "strict" else "0"
    os.environ["MLSBENCH_SYNTAX_GATE"] = "1" if mode == "new" else "0"


def replay(actions, fx, mode):
    setmode(mode)
    tmpd = Path(tempfile.mkdtemp(prefix="rp_"))
    p = tmpd / Path(fx["filename"]).name
    p.write_text(fx["text"])
    t = StubTools(fx["filename"], p, fx["editable"])
    c = collections.Counter()
    try:
        for name, args in actions:
            if name not in ("edit", "undo", "view"):
                continue
            try:
                if name == "edit":
                    r = str(t.edit(**{k: v for k, v in args.items()
                                      if k in ("op", "filename", "content", "after_line",
                                               "start_line", "end_line", "old_str", "new_str")}))
                elif name == "undo":
                    r = str(t.undo(**{k: v for k, v in args.items() if k == "n"}))
                else:
                    r = str(t.view(**{k: v for k, v in args.items()
                                      if k in ("filename", "start_line", "end_line")}))
            except Exception as e:
                r = f"ERROR: harness exception {type(e).__name__}: {e}"
                c["exception"] += 1
            ok = not r.startswith("ERROR")
            c[f"{name}_ok" if ok else f"{name}_err"] += 1
            if name == "edit" and not ok:
                if "REJECTED and NOT applied" in r:
                    c["e_syntax_gate"] += 1
                elif "not found" in r:
                    c["e_notfound"] += 1
                elif "ambiguous" in r or "not unique" in r:
                    c["e_ambig"] += 1
                elif "editable range" in r or "editable region" in r:
                    c["e_range"] += 1
                elif "allow_create" in r:
                    c["e_create"] += 1
                else:
                    c["e_other"] += 1
        src = p.read_text()
        try:
            ast.parse(src)
            c["final_parses"] += 1
        except Exception:
            c["final_broken"] += 1
        c["runs"] += 1
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    return c


tot = {m: collections.Counter() for m in ("strict", "landed", "new")}
nruns = 0
for mp in sorted(LOGS.glob("*/*/*/agent/messages.jsonl")):
    task = mp.parts[mp.parts.index("logs") + 1]
    if task not in fixtures:
        continue
    try:
        recs = [json.loads(l) for l in open(mp) if l.strip()]
    except Exception:
        continue
    acts = [(r["tool_name"], r.get("tool_input") or {}) for r in recs
            if r.get("role") == "assistant"
            and r.get("tool_name") in ("edit", "undo", "view")]
    if not any(a[1].get("op") == "str_replace" for a in acts):
        continue
    nruns += 1
    if nruns % 200 == 0:
        print(f"  {nruns} runs", file=sys.stderr, flush=True)
    for m in ("strict", "landed", "new"):
        tot[m] += replay(acts, fixtures[task], m)

print(f"\nreplayed {nruns} banked transcripts "
      f"(str_replace runs on the {len(fixtures)} tasks with a pristine fixture)\n")
for m in ("strict", "landed", "new"):
    c = tot[m]
    e = c["edit_ok"] + c["edit_err"]
    runs = max(1, c["runs"])
    print(f"  {m:7s} edit calls {e:6d}  accepted {c['edit_ok']:6d} "
          f"({100 * c['edit_ok'] / max(1, e):5.1f}%)  rejected {c['edit_err']:6d} "
          f"({100 * c['edit_err'] / max(1, e):5.1f}%)")
    print(f"          rejects: notfound={c['e_notfound']} ambiguous={c['e_ambig']} "
          f"out_of_range={c['e_range']} syntax_gate={c['e_syntax_gate']} "
          f"create={c['e_create']} other={c['e_other']}")
    print(f"          final file parses in {c['final_parses']}/{runs} runs "
          f"({100 * c['final_parses'] / runs:.1f}%)   undo_err={c['undo_err']}\n")

json.dump({k: dict(v) for k, v in tot.items()},
          open(HERE / "replay_traj_new.json", "w"), indent=1)

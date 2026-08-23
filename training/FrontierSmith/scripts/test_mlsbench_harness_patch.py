#!/usr/bin/env python3
"""Regression tests for the str_replace patch: the loosened matcher must never
retarget an edit, and every existing rejection must still reject."""
import sys, os, tempfile, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scratch_repo"))
from mlsbench.agent.tools import resolve_old_str
from ab_run import StubTools, strict_env

FILE = """\
import numpy as np


class Model:
    def fit(self, X):
        y = X.mean()
        return y

    def predict(self, X):
        y = X.mean()
        return y
"""

fails = []
def check(name, cond, extra=""):
    (print(f"  ok   {name}") if cond else (print(f"  FAIL {name} {extra}"), fails.append(name)))

print("resolve_old_str:")
strict_env(False)
# 1 exact still exact
r = resolve_old_str(FILE, "    def fit(self, X):")
check("exact unique match", r["ok"] and r["level"] == "exact" and r["line_start"] == 5)
# 2 genuinely ambiguous text is still REJECTED, not silently picked
r = resolve_old_str(FILE, "        y = X.mean()\n        return y")
check("duplicate block rejected as ambiguous", (not r["ok"]) and r["reason"] == "ambiguous", r)
# 3 whitespace-tolerant match only when unique
r = resolve_old_str(FILE, "  def fit(self, X):\n      y = X.mean()")
check("ws-tolerant unique match accepted", r["ok"] and r["level"] == "whitespace-insensitive", r)
r = resolve_old_str(FILE, "   y = X.mean()\n   return y")
check("ws-tolerant AMBIGUOUS still rejected", (not r["ok"]) and r["reason"] == "ambiguous", r)
# 4 line-number prefixes
r = resolve_old_str(FILE, "     5:     def fit(self, X):")
check("line-number prefix stripped", r["ok"] and r["line_start"] == 5, r)
# 5 hallucinated text still rejected
r = resolve_old_str(FILE, "    def transform(self, X):\n        return X * 2")
check("absent text still rejected", (not r["ok"]) and r["reason"] == "not_found", r)
# 6 kill-switch reproduces byte-exact behaviour.
#    NB: "  def fit..." would still match byte-exactly, because the harness's
#    matcher is a raw substring search -- an under-indented anchor matches
#    mid-line. That is pre-existing behaviour (4.0% of accepted str_replace edits
#    in the banked runs spliced mid-line); use an anchor that is not a substring.
STRICT_CASE = "    def fit(self,  X):"          # doubled internal space
strict_env(True)
r = resolve_old_str(FILE, STRICT_CASE)
check("STRICT kill-switch rejects ws-normalised match", not r["ok"], r)
strict_env(False)
r = resolve_old_str(FILE, STRICT_CASE)
check("patched matcher accepts the same anchor", r["ok"] and r["line_start"] == 5, r)

print("\nend-to-end via WorkspaceTools.edit:")
tmp = Path(tempfile.mkdtemp()); p = tmp / "m.py"; p.write_text(FILE)
t = StubTools("pkg/m.py", p, [(4, 11)])
# out-of-editable-range still rejected (import line is protected)
out = t.dispatch("edit", {"op": "str_replace", "filename": "pkg/m.py",
                          "old_str": "import numpy as np", "new_str": "import numpy"})
check("protected line still rejected", "exceed the editable range" in out, out[:90])
check("file untouched after rejection", p.read_text() == FILE)
# ws-tolerant accept re-indents new_str to the file
out = t.dispatch("edit", {"op": "str_replace", "filename": "pkg/m.py",
                          "old_str": "  def fit(self, X):\n      y = X.mean()",
                          "new_str": "  def fit(self, X):\n      y = X.max()"})
now = p.read_text()
check("ws-tolerant edit applied", out.startswith("OK"), out[:90])
check("indentation preserved (4 spaces, not 2)", "    def fit(self, X):\n        y = X.max()" in now,
      repr(now[:160]))
check("rest of file intact", "    def predict(self, X):" in now and now.count("import numpy as np") == 1)
check("file still parses", __import__("ast").parse(now) is not None)
# syntax gate fires
out = t.dispatch("edit", {"op": "str_replace", "filename": "pkg/m.py",
                          "old_str": "    def predict(self, X):", "new_str": "    def predict(self, X:"})
check("syntax gate warns on broken file", "no longer parses as Python" in out, out[:120])
# undo now returns the file state
out = t.dispatch("undo", {"n": 1})
check("undo returns a file snapshot", "[Current file:" in out, out[:120])
out = t.dispatch("undo", {"n": 9})
out = t.dispatch("undo", {"n": 1})
check("empty undo tells the model what to do instead", "view()" in out, out[:120])
# view works
out = t.dispatch("view", {"filename": "pkg/m.py", "start_line": 4, "end_line": 6})
check("view returns numbered lines + warning", "display only" in out and "     4: " in out, out[:160])

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)

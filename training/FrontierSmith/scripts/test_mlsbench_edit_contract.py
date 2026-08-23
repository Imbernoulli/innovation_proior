#!/usr/bin/env python3
"""Unit tests for the MLS-Bench agent file-editing contract.

Every test here corresponds to a failure mode measured in the 2819 banked agent
runs under /scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/logs (12255 str_replace calls,
8160 rejected). The tests drive the REAL WorkspaceTools edit path, not a
reimplementation, via a stub that supplies only the workspace plumbing.

Run:  python -m pytest test_edit_contract.py -q
      (or: python test_edit_contract.py  for a dependency-free run)
"""
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

# Which mlsbench source tree to test. Override with MLSBENCH_SRC when validating
# a scratch copy before landing; otherwise this tests the live checkout.
_CANDIDATES = [
    os.environ.get("MLSBENCH_SRC"),
    str(Path(__file__).parent / "repo"),
    "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/src",
]
for _c in _CANDIDATES:
    if _c and (Path(_c) / "mlsbench" / "agent" / "tools.py").exists():
        REPO = Path(_c)
        break
else:  # pragma: no cover
    raise SystemExit(
        "cannot find an mlsbench source tree; set MLSBENCH_SRC=/path/to/src")
sys.path.insert(0, str(REPO))

from mlsbench.agent.tools import (  # noqa: E402
    EDIT_REPLACE_SCHEMA,
    EDIT_REWRITE_SCHEMA,
    VIEW_SCHEMA,
    WorkspaceTools,
    resolve_old_str,
)


# ---------------------------------------------------------------------------
# Fixture: a file shaped like a real MLS-Bench task -- a protected header, a
# bounded editable stub, and a protected footer.
# ---------------------------------------------------------------------------

FILE = textwrap.dedent('''\
    import numpy as np

    # =====================================================================
    # FIXED SECTION -- do not edit
    # =====================================================================
    def helper(x):
        return x * 2

    # =====================================================================
    # EDITABLE: implement solve below
    # =====================================================================
    def solve(X):
        """Solve the task.

        Input:  X of shape (n, d)
        Output: array of shape (n,)
        """
        n = X.shape[0]
        return np.zeros(n)
    # =====================================================================

    def _scoring_entrypoint(X):
        return solve(X)
''')

# 1-indexed: lines 12..19 are the `def solve` body (the editable region).
EDITABLE = [(12, 19)]
FNAME = "numpy/custom_solver.py"


class StubTools(WorkspaceTools):
    """Real edit/view/undo code paths, minimal state (no containers, no tests)."""

    def __init__(self, path: Path, editable=EDITABLE, filename=FNAME,
                 use_replace=True, allow_rewrite=True):
        self._fn = filename
        self._path = path
        self.config_edit = [{
            "filename": filename,
            "edit": [{"start": s, "end": e} for s, e in editable],
        }]
        self.config_task = {"allow_create": False}
        self.all_external_packages = [filename.split("/")[0]]
        self.live_protected_ranges = {
            filename: self._allowed_to_protected([[s, e] for s, e in editable])
        }
        self._history = []
        self._created_files = set()
        self.step_count = 0
        self.use_replace = use_replace
        self.allow_rewrite = allow_rewrite
        self.syntax_gate = True

    def _resolve_workspace_path(self, filename):
        return self._path

    def _find_workspace_pkg(self, pkg):
        raise FileNotFoundError(pkg)


class Base(unittest.TestCase):
    editable = EDITABLE
    body = FILE

    def setUp(self):
        os.environ.pop("MLSBENCH_STRICT_STR_REPLACE", None)
        os.environ["MLSBENCH_SYNTAX_GATE"] = "1"
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "custom_solver.py"
        self.path.write_text(self.body)
        self.t = StubTools(self.path, editable=self.editable)

    def tearDown(self):
        self.tmp.cleanup()

    # helpers
    def text(self):
        return self.path.read_text()

    def edit(self, **kw):
        kw.setdefault("filename", FNAME)
        return self.t.edit(**kw)

    def assertOK(self, res, msg=""):
        self.assertFalse(res.startswith("ERROR"), f"{msg}\n--- got ---\n{res[:900]}")

    def assertERR(self, res, msg=""):
        self.assertTrue(res.startswith("ERROR"), f"{msg}\n--- got ---\n{res[:900]}")


# ===========================================================================
# 1. Stale anchor after an edit
#    Measured: str_replace acceptance decays 50.4% (1st call) -> 40.2% -> 34.9%
#    -> 31.7% -> 31.5% -> 24.7% (6th+), 'old_str not found' dominating from the
#    2nd call on, because the only full copy of the file was in the initial
#    prompt and the post-edit echo showed 3 lines per range.
# ===========================================================================

class TestStaleAnchorAfterEdit(Base):

    def test_echo_contains_the_whole_editable_region_after_an_edit(self):
        res = self.edit(op="str_replace",
                        old_str="    return np.zeros(n)",
                        new_str="    return np.ones(n)")
        self.assertOK(res)
        # The post-edit echo must render the region the agent will anchor on next.
        self.assertIn("def solve(X):", res)
        self.assertIn("return np.ones(n)", res)
        self.assertIn("Current contents of", res)
        for ln in range(12, 20):
            self.assertIn(f"{ln:6d}: ", res, f"line {ln} missing from post-edit echo")

    def test_echo_marks_itself_as_the_source_of_truth(self):
        res = self.edit(op="str_replace",
                        old_str="    n = X.shape[0]",
                        new_str="    n = len(X)")
        self.assertIn("as it is NOW", res)
        self.assertIn("not the copy in the first message", res)

    def test_anchor_taken_from_the_echo_matches_on_the_next_call(self):
        r1 = self.edit(op="str_replace",
                       old_str="    return np.zeros(n)",
                       new_str="    acc = np.zeros(n)\n    return acc")
        self.assertOK(r1)
        # Pull a line straight out of the echo, strip the display prefix, reuse it.
        echoed = [l for l in r1.split("\n") if "acc = np.zeros" in l][-1]
        anchor = echoed.split(": ", 1)[1]
        r2 = self.edit(op="str_replace", old_str=anchor,
                       new_str="    acc = np.full(n, 0.5)")
        self.assertOK(r2, "an anchor copied out of the echo must match")
        self.assertIn("np.full(n, 0.5)", self.text())

    def test_stale_anchor_from_the_original_prompt_is_reported_with_live_text(self):
        self.assertOK(self.edit(op="rewrite",
                                content="def solve(X):\n    return X.sum(axis=1)\n"))
        # Now quote the ORIGINAL stub, which no longer exists.
        res = self.edit(op="str_replace",
                        old_str="    n = X.shape[0]\n    return np.zeros(n)",
                        new_str="    return X.mean(1)")
        self.assertERR(res)
        self.assertIn("not found", res)
        # The error must show the file as it stands, so the retry can succeed.
        self.assertIn("return X.sum(axis=1)", res)


# ===========================================================================
# 2. Whitespace / indentation drift   (banked class E: 628 cases)
# ===========================================================================

class TestWhitespaceDrift(Base):

    def test_wrong_indentation_still_matches_when_unique(self):
        res = self.edit(op="str_replace",
                        old_str="        n = X.shape[0]\n        return np.zeros(n)",  # 8 not 4
                        new_str="        return X.sum(1)")
        self.assertOK(res, "indent-only drift must not reject a unique anchor")
        self.assertIn("whitespace", res.lower())

    def test_reindent_preserves_the_file_indentation(self):
        self.assertOK(self.edit(
            op="str_replace",
            old_str="        n = X.shape[0]\n        return np.zeros(n)",
            new_str="        total = X.sum(1)\n        return total"))
        self.assertIn("    total = X.sum(1)", self.text())
        self.assertIn("    return total", self.text())
        import ast
        ast.parse(self.text())

    def test_trailing_whitespace_is_tolerated(self):
        res = self.edit(op="str_replace",
                        old_str="    return np.zeros(n)   ",
                        new_str="    return np.ones(n)")
        self.assertOK(res)

    def test_collapsed_internal_spacing_is_tolerated(self):
        res = self.edit(op="str_replace",
                        old_str="    n =  X.shape[0]",
                        new_str="    n = int(X.shape[0])")
        self.assertOK(res)


# ===========================================================================
# 3. Line-number-prefix contamination  (banked class B: 132 cases)
#    Every listing the agent sees is rendered "%6d: <line>".
# ===========================================================================

class TestLineNumberPrefix(Base):

    def test_prompt_style_colon_prefix_is_stripped(self):
        res = self.edit(
            op="str_replace",
            old_str="    18:     n = X.shape[0]\n    19:     return np.zeros(n)",
            new_str="    return X.sum(1)")
        self.assertOK(res, "the 'NNNNNN: ' display prefix must not break the match")
        self.assertNotIn("18:", self.text())

    def test_snapshot_style_prefix_is_stripped(self):
        res = self.edit(op="str_replace",
                        old_str="    18      n = X.shape[0]",
                        new_str="    n = len(X)")
        self.assertOK(res)

    def test_view_and_echo_use_the_same_prefix_format(self):
        v = self.t.view(FNAME)
        e = self.edit(op="str_replace", old_str="    n = X.shape[0]",
                      new_str="    n = len(X)")
        self.assertIn(f"{12:6d}: ", v)
        self.assertIn(f"{12:6d}: ", e)

    def test_every_listing_warns_against_including_the_prefix(self):
        self.assertIn("NNNNNN", self.t.view(FNAME))
        self.assertIn("NNNNNN",
                      self.edit(op="str_replace", old_str="    n = X.shape[0]",
                                new_str="    n = len(X)"))


# ===========================================================================
# 4. Mid-line splice
#    A raw substring hit that is not on line boundaries corrupts the file;
#    banked mid-line matches left it unparseable 49.6% of the time.
# ===========================================================================

class TestMidLineSplice(Base):

    def test_under_indented_anchor_snaps_to_line_boundaries(self):
        res = self.edit(op="str_replace",
                        old_str="return np.zeros(n)",
                        new_str="return np.ones(n)")
        self.assertOK(res)
        import ast
        ast.parse(self.text())
        self.assertIn("    return np.ones(n)", self.text())
        self.assertNotIn("np.ones(n))", self.text())

    def test_no_orphan_fragment_left_behind(self):
        self.assertOK(self.edit(op="str_replace",
                                old_str="n = X.shape[0]",
                                new_str="n = len(X)"))
        for line in self.text().split("\n"):
            s = line.strip()
            self.assertFalse(s and not s.startswith("#") and s.endswith("        "),
                             f"orphan indentation fragment: {line!r}")
        import ast
        ast.parse(self.text())


# ===========================================================================
# 5. Out of editable range
#    Measured: 316 of the 1913 FIRST str_replace calls in the banked runs died
#    here -- the anchor starts in the protected header and runs into the body.
# ===========================================================================

class TestOutOfEditableRange(Base):

    def test_anchor_spanning_the_protected_header_is_rejected(self):
        res = self.edit(
            op="str_replace",
            old_str="# EDITABLE: implement solve below\n"
                    "# =====================================================================\n"
                    "def solve(X):",
            new_str="def solve(X):")
        self.assertERR(res)
        self.assertIn("outside the editable region", res)

    def test_the_error_names_both_ranges_and_says_what_to_do(self):
        res = self.edit(op="str_replace",
                        old_str="def helper(x):\n    return x * 2",
                        new_str="def helper(x):\n    return x * 3")
        self.assertERR(res)
        self.assertIn("Editable:", res)
        self.assertIn("protected:", res)
        self.assertIn("Shorten old_str", res)

    def test_the_error_offers_rewrite(self):
        res = self.edit(
            op="str_replace",
            old_str="# =====================================================================\n"
                    "def solve(X):",
            new_str="def solve(X):")
        self.assertERR(res)
        self.assertIn("op='rewrite'", res)

    def test_a_rejected_out_of_range_edit_changes_nothing(self):
        before = self.text()
        self.edit(op="str_replace", old_str="def helper(x):\n    return x * 2",
                  new_str="BROKEN")
        self.assertEqual(before, self.text())

    def test_rewrite_can_never_leave_the_editable_region(self):
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return X\n"))
        self.assertIn("def helper(x):", self.text())
        self.assertIn("FIXED SECTION", self.text())
        self.assertIn("_scoring_entrypoint", self.text())


# ===========================================================================
# 6. Non-unique anchor
# ===========================================================================

class TestNonUniqueAnchor(Base):
    body = FILE.replace(
        "    n = X.shape[0]\n    return np.zeros(n)",
        "    n = X.shape[0]\n    y = np.zeros(n)\n    y = np.zeros(n)\n    return y")
    editable = [(12, 21)]

    def test_duplicate_anchor_is_rejected_not_guessed(self):
        before = self.text()
        res = self.edit(op="str_replace", old_str="    y = np.zeros(n)",
                        new_str="    y = np.ones(n)")
        self.assertERR(res)
        self.assertIn("ambiguous", res)
        self.assertEqual(before, self.text(), "an ambiguous edit must not be applied")

    def test_the_error_lists_where_the_duplicates_are(self):
        res = self.edit(op="str_replace", old_str="    y = np.zeros(n)",
                        new_str="    y = np.ones(n)")
        self.assertIn("starting at lines", res)
        self.assertIn("2 places", res)

    def test_extending_the_anchor_disambiguates(self):
        res = self.edit(
            op="str_replace",
            old_str="    y = np.zeros(n)\n    return y",
            new_str="    y = np.ones(n)\n    return y")
        self.assertOK(res)

    def test_a_loose_level_never_overrides_a_unique_stricter_match(self):
        # Exact-unique must win even though a whitespace-insensitive search would
        # find two candidates.
        r = resolve_old_str("if x:\n    a = 1\nif x:\n     a = 1\n", "    a = 1")
        self.assertFalse(r["ok"])  # 2 exact-insensitive candidates -> refuse


# ===========================================================================
# 7. Post-undo staleness
#    Measured: str_replace immediately after undo() was accepted 34.8% of the
#    time (925 of 1820 rejections were 'old_str not found') -- undo rewound the
#    file and returned no content at all.
# ===========================================================================

class TestPostUndoStaleness(Base):

    def test_undo_echoes_the_restored_file(self):
        self.assertOK(self.edit(op="rewrite",
                                content="def solve(X):\n    return X.sum(1)\n"))
        res = self.t.undo()
        self.assertIn("Undo complete", res)
        self.assertIn("Current contents of", res)
        self.assertIn("return np.zeros(n)", res)
        self.assertNotIn("return X.sum(1)", res)

    def test_an_anchor_read_off_the_undo_echo_matches(self):
        self.assertOK(self.edit(op="rewrite",
                                content="def solve(X):\n    return X.sum(1)\n"))
        undo_out = self.t.undo()
        line = [l for l in undo_out.split("\n") if "np.zeros(n)" in l][-1]
        anchor = line.split(": ", 1)[1]
        self.assertOK(self.edit(op="str_replace", old_str=anchor,
                                new_str="    return np.ones(n)"))

    def test_undo_with_empty_history_is_actionable(self):
        res = self.t.undo()
        self.assertERR(res)
        self.assertIn("Do not call undo again", res)

    def test_undo_restores_the_editable_ranges_too(self):
        before = self.t._editable_range_str(FNAME)
        self.assertOK(self.edit(
            op="rewrite",
            content="def solve(X):\n" + "    pass\n" * 40))
        self.t.undo()
        self.assertEqual(before, self.t._editable_range_str(FNAME))


# ===========================================================================
# 8. Syntax-breaking edit
#    Measured: runs where the editable file always parsed scored 0.0836; runs
#    where it broke at least once scored 0.0559.
# ===========================================================================

class TestSyntaxGate(Base):

    def test_an_edit_that_breaks_the_parse_is_rejected_and_reverted(self):
        before = self.text()
        res = self.edit(op="rewrite", content="def solve(X):\n    return (X.sum(\n")
        self.assertERR(res)
        self.assertIn("REJECTED and NOT applied", res)
        self.assertEqual(before, self.text(), "the file must be left untouched")

    def test_the_rejection_names_the_parse_error_and_the_line(self):
        res = self.edit(op="rewrite", content="def solve(X):\n    return (X.sum(\n")
        self.assertIn("unparseable", res)
        self.assertTrue("SyntaxError" in res or "IndentationError" in res, res[:400])
        self.assertIn("Where it would have broken", res)

    def test_a_valid_edit_still_applies(self):
        self.assertOK(self.edit(op="rewrite",
                                content="def solve(X):\n    return X.sum(1)\n"))
        self.assertIn("return X.sum(1)", self.text())

    def test_the_gate_steps_aside_after_repeated_failures(self):
        bad = "def solve(X):\n    return (X.sum(\n"
        for i in range(WorkspaceTools.SYNTAX_GATE_MAX_STREAK):
            self.assertERR(self.edit(op="rewrite", content=bad), f"attempt {i}")
        # Nth+1 must go through, so a model that cannot satisfy the gate is not
        # locked out of editing for the rest of its 20-step budget.
        res = self.edit(op="rewrite", content=bad)
        self.assertOK(res)
        self.assertIn("WARNING", res)

    def test_a_good_edit_resets_the_streak(self):
        bad = "def solve(X):\n    return (X.sum(\n"
        self.assertERR(self.edit(op="rewrite", content=bad))
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return X\n"))
        self.assertEqual(self.t._syntax_reject_streak.get(FNAME, 0), 0)

    def test_an_already_broken_file_is_not_held_against_the_next_edit(self):
        os.environ["MLSBENCH_SYNTAX_GATE"] = "0"
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return (\n"))
        os.environ["MLSBENCH_SYNTAX_GATE"] = "1"
        # File is already unparseable; an edit that leaves it unparseable must
        # still be allowed through (only NEW breakage is rejected).
        res = self.edit(op="rewrite", content="def solve(X):\n    return ((\n")
        self.assertOK(res)

    def test_the_gate_can_be_switched_off(self):
        os.environ["MLSBENCH_SYNTAX_GATE"] = "0"
        res = self.edit(op="rewrite", content="def solve(X):\n    return (X.sum(\n")
        self.assertOK(res)
        self.assertIn("WARNING", res)

    def test_other_drivers_are_not_gated(self):
        # openevolve_agent / discover_agent call tools.edit() directly and never
        # set syntax_gate; their edit semantics must be unchanged.
        before = self.text()
        t = StubTools(self.path)
        t.syntax_gate = False
        res = t.edit(op="rewrite", filename=FNAME,
                     content="def solve(X):\n    return (X.sum(\n")
        self.assertFalse(res.startswith("ERROR"))
        self.assertIn("WARNING", res)
        self.assertNotEqual(before, self.text(), "the edit should have been applied")

    def test_non_python_files_are_not_gated(self):
        p = Path(self.tmp.name) / "conf.yaml"
        p.write_text("a: 1\nb: 2\nc: 3\n")
        t = StubTools(p, editable=[(1, 3)], filename="numpy/conf.yaml")
        self.assertFalse(t.edit(op="rewrite", filename="numpy/conf.yaml",
                                content="a: [\n").startswith("ERROR"))


# ===========================================================================
# 9. op='rewrite' -- the new anchor-free path
#    Measured: 3098 of 12255 banked str_replace calls used an anchor covering
#    >=95% of the editable region and only 12.1% were accepted, against 37-45%
#    for smaller anchors.
# ===========================================================================

class TestRewriteOp(Base):

    def test_rewrite_replaces_exactly_the_editable_region(self):
        res = self.edit(op="rewrite", content="def solve(X):\n    return X.mean(1)\n")
        self.assertOK(res)
        t = self.text()
        self.assertIn("def solve(X):\n    return X.mean(1)", t)
        self.assertNotIn("np.zeros", t)
        self.assertIn("import numpy as np", t)          # header intact
        self.assertIn("def _scoring_entrypoint", t)     # footer intact
        self.assertIn("return x * 2", t)                # protected helper intact

    def test_rewrite_needs_no_anchor_and_no_line_numbers(self):
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return X\n"))
        # ... and is still fine when the region has already been replaced once.
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return -X\n"))
        self.assertIn("return -X", self.text())

    def test_rewrite_reports_the_lines_it_replaced(self):
        res = self.edit(op="rewrite", content="def solve(X):\n    return X\n")
        self.assertIn("replaced lines 12..19", res)
        self.assertIn("Editable range:", res)

    def test_ranges_track_a_size_change(self):
        self.assertOK(self.edit(op="rewrite",
                                content="def solve(X):\n" + "    z = 1\n" * 20))
        self.assertEqual(self.t._editable_range_str(FNAME), "12–32")
        self.assertIn("def _scoring_entrypoint", self.text())
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return X\n"))
        self.assertEqual(self.t._editable_range_str(FNAME), "12–13")

    def test_rewrite_is_undoable(self):
        before = self.text()
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return X\n"))
        self.t.undo()
        self.assertEqual(before, self.text())

    def test_empty_content_is_refused_not_applied(self):
        before = self.text()
        res = self.edit(op="rewrite", content="")
        self.assertERR(res)
        self.assertIn("needs `content`", res)
        self.assertEqual(before, self.text())

    def test_trailing_newline_is_normalised(self):
        self.assertOK(self.edit(op="rewrite", content="def solve(X):\n    return X"))
        import ast
        ast.parse(self.text())
        self.assertIn("    return X\n# ====", self.text())

    def test_rewrite_refuses_a_multi_region_file_with_an_actionable_error(self):
        t = StubTools(self.path, editable=[(12, 15), (17, 19)])
        res = t.edit(op="rewrite", filename=FNAME, content="whatever\n")
        self.assertERR(res)
        self.assertIn("2 separate editable regions", res)
        self.assertIn("str_replace", res)

    def test_rewrite_on_a_fully_protected_file_is_refused(self):
        t = StubTools(self.path, editable=[])
        res = t.edit(op="rewrite", filename=FNAME, content="x = 1\n")
        self.assertERR(res)
        self.assertIn("no editable region", res)


# ===========================================================================
# 10. Schema / prompt contract and op-name leniency
# ===========================================================================

class TestContractSurface(Base):

    def test_schema_advertises_the_three_ops_with_their_companions(self):
        props = EDIT_REWRITE_SCHEMA["input_schema"]["properties"]
        self.assertEqual(props["op"]["enum"], ["rewrite", "str_replace", "create"])
        d = props["op"]["description"]
        for op in ("rewrite", "str_replace", "create"):
            self.assertIn(op, d)
        self.assertIn("requires content", d)
        self.assertIn("requires old_str", d)

    def test_schema_tells_the_model_when_to_use_which(self):
        d = EDIT_REWRITE_SCHEMA["description"]
        self.assertIn("rewrite", d)
        self.assertIn("1 to 10 lines", d)
        self.assertIn("echoes the file's current editable region", d)

    def test_view_is_part_of_the_contract(self):
        self.assertEqual(VIEW_SCHEMA["name"], "view")
        self.assertEqual(VIEW_SCHEMA["input_schema"]["required"], ["filename"])

    def test_op_synonyms_are_accepted(self):
        for alias in ("REWRITE", "overwrite", "replace_region", " rewrite "):
            self.setUp()
            res = self.edit(op=alias, content="def solve(X):\n    return X\n")
            self.assertOK(res, f"alias {alias!r} should map to rewrite")

    def test_bare_replace_with_only_content_is_treated_as_rewrite(self):
        res = self.edit(op="replace", content="def solve(X):\n    return X\n")
        self.assertOK(res, "op='replace'+content in str_replace mode means rewrite")
        self.assertIn("return X", self.text())

    def test_upstream_schema_is_left_untouched_for_the_strict_arm(self):
        # The strict arm must keep reproducing public MLS-Bench exactly, so the
        # original schema object must not have grown a 'rewrite' op.
        self.assertEqual(
            EDIT_REPLACE_SCHEMA["input_schema"]["properties"]["op"]["enum"],
            ["create", "str_replace"])
        self.assertNotIn("rewrite", EDIT_REPLACE_SCHEMA["description"])

    def test_rewrite_is_refused_when_the_arm_withholds_it(self):
        before = self.text()
        t = StubTools(self.path, allow_rewrite=False)
        res = t.edit(op="rewrite", filename=FNAME, content="def solve(X):\n    return X\n")
        self.assertERR(res)
        self.assertIn("unknown op 'rewrite'", res)
        self.assertIn("str_replace", res)
        self.assertEqual(before, self.text())

    def test_rewrite_withheld_arm_does_not_advertise_rewrite_in_errors(self):
        t = StubTools(self.path, allow_rewrite=False)
        res = t.edit(op="str_replace", filename=FNAME,
                     old_str="nothing like this exists at all",
                     new_str="x")
        self.assertERR(res)
        self.assertNotIn("op='rewrite'", res)

    def test_oversized_anchor_is_steered_to_rewrite(self):
        res = self.edit(op="str_replace",
                        old_str="\n".join(f"bogus line {i}" for i in range(30)),
                        new_str="x")
        self.assertERR(res)
        self.assertIn("op='rewrite'", res)


# ===========================================================================
# 11. Echo budget -- the echo must stay inside the RL loop's tool-response cap
#     (MLS_RL_MAX_TOOL_RESPONSE_CHARS, default 8000).
# ===========================================================================

class TestEchoBudget(Base):

    def test_a_large_region_is_elided_with_a_pointer_to_view(self):
        big = "def solve(X):\n" + "".join(f"    z{i} = {i}\n" for i in range(400))
        os.environ["MLSBENCH_SYNTAX_GATE"] = "0"
        res = self.edit(op="rewrite", content=big)
        self.assertOK(res)
        self.assertLessEqual(len(res), WorkspaceTools.SNAPSHOT_MAX_CHARS + 1200)
        self.assertIn("view(", res)

    def test_a_normal_region_is_echoed_whole(self):
        res = self.edit(op="rewrite",
                        content="def solve(X):\n" + "".join(
                            f"    z{i} = {i}\n" for i in range(40)))
        self.assertOK(res)
        self.assertIn("z0 = 0", res)
        self.assertIn("z39 = 39", res)
        self.assertNotIn("more lines not shown", res)

    def test_echo_fits_the_rl_tool_response_cap(self):
        big = "def solve(X):\n" + "".join(f"    z{i} = {i}\n" for i in range(400))
        os.environ["MLSBENCH_SYNTAX_GATE"] = "0"
        self.edit(op="rewrite", content=big)
        self.assertLess(len(self.t._file_snapshot(FNAME)), 8000)


# ===========================================================================
# 12. view()
# ===========================================================================

class TestView(Base):

    def test_view_defaults_to_the_editable_region(self):
        v = self.t.view(FNAME)
        self.assertIn("def solve(X):", v)
        self.assertIn("editable:", v)

    def test_view_honours_an_explicit_range(self):
        v = self.t.view(FNAME, start_line=1, end_line=2)
        self.assertIn("import numpy", v)
        self.assertNotIn("def solve", v)

    def test_view_reflects_edits(self):
        self.edit(op="rewrite", content="def solve(X):\n    return X.std(1)\n")
        self.assertIn("X.std(1)", self.t.view(FNAME))

    def test_view_rejects_an_empty_range(self):
        self.assertERR(self.t.view(FNAME, start_line=9, end_line=3))

    def test_view_rejects_a_missing_file(self):
        p = Path(self.tmp.name) / "gone.py"
        t = StubTools(p, editable=[(1, 1)], filename="numpy/gone.py")
        self.assertERR(t.view("numpy/gone.py"))

    def test_view_clamps_past_eof_instead_of_crashing(self):
        v = self.t.view(FNAME, start_line=1, end_line=10_000)
        self.assertIn("return solve(X)", v)
        self.assertFalse(v.startswith("ERROR"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ===========================================================================
# 13. Arm selection in interactive.py — the contract must compose with the
#     MLSBENCH_STRICT_STR_REPLACE / MLSBENCH_VIEW_TOOL A/B knobs.
# ===========================================================================

class TestArmSelection(unittest.TestCase):
    """Drives the real InteractiveAgent.__init__ body with BaseAgent stubbed out."""

    def _build(self, **env):
        import importlib
        import mlsbench.agent.interactive as I
        for k in ("MLSBENCH_STRICT_STR_REPLACE", "MLSBENCH_VIEW_TOOL", "MLSBENCH_REWRITE_OP"):
            os.environ.pop(k, None)
        os.environ.update(env)
        importlib.reload(I)

        class FakeTools:
            pass

        class Agent(I.InteractiveAgent):
            def __init__(self):
                self.tools = FakeTools()
                self.task_name = "t"
                I.BaseAgent.__init__ = lambda *a, **k: None
                I.build_client = lambda *a, **k: None
                I.InteractiveAgent.__init__(self, "t", {"use_replace": True}, None)

        a = Agent()
        names = [s["name"] for s in a._tool_schemas]
        edit = next(s for s in a._tool_schemas if s["name"] == "edit")
        return a, names, edit

    def tearDown(self):
        for k in ("MLSBENCH_STRICT_STR_REPLACE", "MLSBENCH_VIEW_TOOL", "MLSBENCH_REWRITE_OP"):
            os.environ.pop(k, None)

    def test_default_arm_gets_rewrite_and_view(self):
        a, names, edit = self._build()
        self.assertIn("view", names)
        self.assertEqual(edit["input_schema"]["properties"]["op"]["enum"],
                         ["rewrite", "str_replace", "create"])
        self.assertTrue(a.tools.allow_rewrite)
        self.assertIn("op='rewrite'", a.system_prompt)

    def test_default_arm_enables_the_syntax_gate(self):
        a, _, _ = self._build()
        self.assertTrue(a.tools.syntax_gate)

    def test_strict_arm_is_upstream_exactly(self):
        a, names, edit = self._build(MLSBENCH_STRICT_STR_REPLACE="1")
        self.assertFalse(a.tools.syntax_gate)
        self.assertNotIn("view", names)
        self.assertEqual(edit["input_schema"]["properties"]["op"]["enum"],
                         ["create", "str_replace"])
        self.assertFalse(getattr(a.tools, "allow_rewrite", False))
        self.assertNotIn("rewrite", a.system_prompt)
        self.assertIn("must match exactly — whitespace included", a.system_prompt)

    def test_rewrite_can_be_withheld_without_losing_view(self):
        a, names, edit = self._build(MLSBENCH_REWRITE_OP="0")
        self.assertIn("view", names)
        self.assertEqual(edit["input_schema"]["properties"]["op"]["enum"],
                         ["create", "str_replace"])
        self.assertFalse(getattr(a.tools, "allow_rewrite", False))
        self.assertNotIn("op='rewrite'", a.system_prompt)

    def test_view_can_be_withheld_while_rewrite_stays(self):
        a, names, edit = self._build(MLSBENCH_VIEW_TOOL="0")
        self.assertNotIn("view", names)
        self.assertEqual(edit["input_schema"]["properties"]["op"]["enum"],
                         ["rewrite", "str_replace", "create"])
        self.assertIn("op='rewrite'", a.system_prompt)

#!/usr/bin/env python3
"""Reference-grounded innovation metric for MLS-Bench agent episodes.

WHAT THIS MEASURES
------------------
"Innovation" here is *distance from the references the task itself ships*, not
keyword counting.  Every MLS-Bench task provides three ground-truth anchors:

  1. ``tasks/<t>/edits/custom_template.py`` -- the stub the agent starts from,
     with the editable line range declared in ``tasks/<t>/config.json``
     (``files[].edit``).
  2. ``tasks/<t>/config.json:baselines`` -- named ``baseline:*`` methods, each a
     real edit-op file under ``tasks/<t>/edits/*.edit.py`` whose ``OPS`` rewrite
     the same editable region.  These are the *standard approaches*.
  3. ``tasks/<t>/leaderboard.csv`` + ``score_spec.py`` -- the score each named
     baseline actually achieves, so "beat the best baseline" is checkable.

So for one episode we can materialise, in the *same* representation:
    T   = template editable region
    B_k = editable region as baseline k would have written it
    S   = editable region the agent actually submitted
and place S on a ladder between T and "different from every B_k".

LEVELS (per episode)
--------------------
  BROKEN  submission does not parse, or dropped a symbol the task contract
          requires (a class/def/method that the template defines).  Kept out of
          the innovation axis: destroying the stub is not a new method.
  L0      unmodified template          (byte-identical editable region)
  L1      cosmetic edit                (same AST after dropping docstrings/
                                        comments/formatting, or after
                                        alpha-renaming locals)
  L2      parameter change             (identical AST skeleton once every
                                        numeric/string constant is blanked)
  L3      same algorithm family as a named baseline
  L4      different algorithm from every named baseline
  L5      L4 *and* the episode's task score beats the best named baseline

  CONSERVATIVE = {L0,L1,L2,L3}   INNOVATIVE = {L4,L5}

REPRESENTATIONS
---------------
Structural, not textual.  Four canonical forms of a code region, each obtained
by parsing with ``ast`` and re-emitting with ``ast.unparse`` (so whitespace,
line breaks, quote style and comments cannot register as a change):

    ast_norm   docstrings stripped                     -> L1 test vs template
    ast_alpha  + locals alpha-renamed v0,v1,...         -> L1 test vs template
    skeleton   + every str/num constant blanked         -> L2 test
    opbag      weighted multiset of *semantic* features -> L3 family test

``opbag`` features are dotted call targets (``np.argsort``,
``self.get_grad_embedding``, ``KMeans``), imported symbols, attribute names,
control-flow node types and operators.  Crucially each feature is scored only
if it is *discriminative for this task*: features already present in the
template are subtracted first.  That is the reference-grounded replacement for
a global lexicon -- the task tells us what its own boilerplate looks like, so
``np.arange`` or ``self.idxs_lb`` can never count as evidence of a method.

FAMILY MATCH (L3) decision rule, first hit wins:
    replica     ast_alpha(S) == ast_alpha(B_k)
    retuned     skeleton(S)  == skeleton(B_k)
    structural  jaccard(S,B_k)    >= TAU_HIGH
    subsumed    subsumption(S,B_k) >= TAU_SUB   (S adds no vocabulary beyond B_k)
    declared    jaccard(S,B_k)    >= TAU_LOW  and  S's prose/identifiers name B_k

``jaccard`` is weighted Jaccard over discriminative features; ``subsumption`` is
the share of S's own discriminative weight that B_k also covers.  Both are
needed: Jaccard catches "re-implemented BADGE", subsumption catches the thin
wrapper -- ``DBSCAN(...)`` with a KMeans fallback has low Jaccard against the
much richer ``baseline:dbscan`` edit yet introduces nothing new, and calling it
a different algorithm would be exactly the kind of false positive this metric
exists to avoid.

Both thresholds are calibrated, not chosen by taste: ``--calibrate`` scores every
pair of *distinct* baselines within a task -- known different algorithms -- and
TAU_HIGH / TAU_SUB sit just above the 95th percentile of those distributions
(Jaccard p95=0.52 -> TAU_HIGH=0.55, 5/140 pairs flagged; subsumption p95=0.81 ->
TAU_SUB=0.85, 13/280 flagged, and the flagged pairs are genuinely nested families
like xgboost_style/gradient_boosting and flat_zigzag/zigzag).  Baseline aliases
come from the baseline's own key and its edit-file docstring -- again task-local,
not a global vocabulary.

SCORE AXIS
----------
Two reference points, both reported:
    stub score          -- what submitting the untouched template scores.
                           Taken from the empirical median over L0/L1 episodes
                           in the corpus being scanned, falling back to the
                           bundled table (``--stub-scores``).
    best baseline score -- max over ``baseline:*`` rows of leaderboard.csv,
                           scored through the task's own score_spec.py.
The 2x2 the metric reports is innovation x (score > stub score):
    innovative-and-better / innovative-but-worse
    conservative-and-safe / conservative-and-worse

USAGE
-----
    python scripts/mlsbench_innovation_metric.py OUTDIR [OUTDIR ...]
    python scripts/mlsbench_innovation_metric.py --calibrate
    python scripts/mlsbench_innovation_metric.py --self-test

OUTDIR is an MLS-Bench CPU run directory (``outputs/cc_mlsbench_cpu_*``)
containing ``summary.json`` and ``task_logs/``.  The submitted code is read from
the per-episode workspace that the run's log points at
(``vendor/workspace/<task>/<runid>/``).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MLS_ROOT_DEFAULT = Path(
    os.environ.get("MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev")
)
HERE = Path(__file__).resolve().parent
STUB_SCORE_TABLE = HERE / "mlsbench_innovation_stub_scores.json"

# Feature weights for the weighted-Jaccard family similarity.  Calls and
# imports carry algorithm identity; control flow and operators are shape only.
FEATURE_WEIGHTS = {"call": 1.0, "imp": 1.0, "attr": 0.5, "flow": 0.25, "op": 0.25}

TAU_HIGH = 0.55        # structural family match (Jaccard), no naming needed
TAU_LOW = 0.30         # weaker structural match, needs the submission to name the baseline
TAU_SUB = 0.85         # subsumption: submission adds no vocabulary beyond one baseline
MIN_DISC_WEIGHT = 2.0  # below this the submission is too small to judge by subsumption

LEVELS = ["BROKEN", "L0", "L1", "L2", "L3", "L4", "L5"]
LEVEL_LABEL = {
    "BROKEN": "broken / contract violated",
    "L0": "unmodified template",
    "L1": "cosmetic edit",
    "L2": "parameter change",
    "L3": "baseline family",
    "L4": "different algorithm",
    "L5": "different algorithm + beats best baseline",
}
CONSERVATIVE = {"L0", "L1", "L2", "L3"}
INNOVATIVE = {"L4", "L5"}


# ---------------------------------------------------------------------------
# canonical forms
# ---------------------------------------------------------------------------

def _try_parse(src: str) -> ast.Module | None:
    """Parse a code region.  Regions are slices of a file, so retry dedented."""
    for candidate in (src, _dedent_block(src)):
        if candidate is None:
            continue
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
    return None


def _dedent_block(src: str) -> str | None:
    lines = [ln for ln in src.splitlines() if ln.strip()]
    if not lines:
        return None
    indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
    if indent == 0:
        return None
    return "\n".join(ln[indent:] if ln.strip() else ln for ln in src.splitlines())


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)


def _collect_locals(tree: ast.AST) -> set[str]:
    """Names bound inside the region: assignment targets, args, loop/with/comp vars."""
    out: set[str] = set()

    def add_target(t):
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.add(n.id)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign,)):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                add_target(node.optional_vars)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                out.add(arg.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
    # never rename `self`/`cls` (they anchor the contract) or imported modules
    out.discard("self")
    out.discard("cls")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                out.discard((al.asname or al.name).split(".")[0])
    return out


class _AlphaRenamer(ast.NodeTransformer):
    def __init__(self, renamable: set[str]):
        self.renamable = renamable
        self.mapping: dict[str, str] = {}

    def _map(self, name: str) -> str:
        if name not in self.renamable:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v{len(self.mapping)}"
        return self.mapping[name]

    def visit_Name(self, node):
        node.id = self._map(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._map(node.arg)
        node.annotation = None
        return node

    def visit_ExceptHandler(self, node):
        self.generic_visit(node)
        if node.name:
            node.name = self._map(node.name)
        return node


class _ConstantBlanker(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, bool) or node.value is None or node.value is Ellipsis:
            return node
        if isinstance(node.value, (int, float, complex)):
            return ast.Constant(value=0)
        if isinstance(node.value, str):
            return ast.Constant(value="")
        if isinstance(node.value, bytes):
            return ast.Constant(value=b"")
        return node


@dataclass
class Canon:
    """Canonical forms of one code region."""
    src: str
    parsed: bool
    ast_norm: str = ""
    ast_alpha: str = ""
    skeleton: str = ""
    features: Counter = field(default_factory=Counter)
    toplevel: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    prose: str = ""


def canonicalize(src: str) -> Canon:
    tree = _try_parse(src)
    if tree is None:
        return Canon(src=src, parsed=False, prose=_prose_of(src))
    norm_tree = _DocstringStripper().visit(ast.parse(ast.unparse(tree)))
    ast.fix_missing_locations(norm_tree)
    ast_norm = ast.unparse(norm_tree)

    alpha_tree = ast.parse(ast_norm)
    renamable = _collect_locals(alpha_tree)
    alpha_tree = _AlphaRenamer(renamable).visit(alpha_tree)
    ast.fix_missing_locations(alpha_tree)
    ast_alpha = ast.unparse(alpha_tree)

    skel_tree = _ConstantBlanker().visit(ast.parse(ast_alpha))
    ast.fix_missing_locations(skel_tree)
    skeleton = ast.unparse(skel_tree)

    return Canon(
        src=src,
        parsed=True,
        ast_norm=ast_norm,
        ast_alpha=ast_alpha,
        skeleton=skeleton,
        features=extract_features(tree),
        toplevel={n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))},
        symbols=collect_symbols(tree),
        prose=_prose_of(src),
    )


def _prose_of(src: str) -> str:
    """Text in which a submission can *declare* what method it implements.

    Comments, string literals, and identifiers that are NOT attribute accesses.
    Excluding ``x.attr`` is what stops ``np.random.permutation`` from counting as
    the submission naming ``baseline:random`` -- the same class of false positive
    (``pid`` matching a Triton program id) that made the lexicon metric noisy.
    """
    import tokenize
    out: list[str] = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r"\.\w+", " ", src).lower()
    prev_op_dot = False
    for tok in toks:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            out.append(tok.string)
            prev_op_dot = False
        elif tok.type == tokenize.NAME:
            if not prev_op_dot:
                out.append(tok.string)
            prev_op_dot = False
        elif tok.type == tokenize.OP:
            prev_op_dot = tok.string == "."
        elif tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue
        else:
            prev_op_dot = False
    return " ".join(out).lower()


def collect_symbols(tree: ast.AST) -> set[str]:
    """Contract symbols: top-level defs/classes and the methods of those classes."""
    out: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(f"{node.name}.{sub.name}")
    return out


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def extract_features(tree: ast.AST) -> Counter:
    feats: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            d = _dotted(node.func)
            if d:
                feats[f"call:{d}"] += 1
        elif isinstance(node, ast.Attribute):
            feats[f"attr:{node.attr}"] += 1
        elif isinstance(node, ast.Import):
            for al in node.names:
                feats[f"imp:{al.name}"] += 1
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                feats[f"imp:{node.module or ''}.{al.name}"] += 1
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try,
                               ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                               ast.Lambda, ast.With, ast.Yield)):
            feats[f"flow:{type(node).__name__}"] += 1
        elif isinstance(node, (ast.BinOp,)):
            feats[f"op:{type(node.op).__name__}"] += 1
        elif isinstance(node, ast.Compare):
            for o in node.ops:
                feats[f"op:{type(o).__name__}"] += 1
    return feats


def _w(feat: str) -> float:
    return FEATURE_WEIGHTS.get(feat.split(":", 1)[0], 0.25)


def wsum(feats: Iterable[str]) -> float:
    return sum(_w(f) for f in feats)


def weighted_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = wsum(a & b)
    union = wsum(a | b)
    return inter / union if union else 0.0


def subsumption(sub_f: set[str], base_f: set[str]) -> float:
    """Fraction of the submission's own method vocabulary that the baseline also has.

    Jaccard alone misses the thin-wrapper case: a submission that just calls
    ``sklearn.cluster.DBSCAN`` with a KMeans fallback has low Jaccard against the
    (much richer) ``baseline:dbscan`` edit, yet it introduces *nothing* the
    baselines do not already do.  Subsumption near 1.0 says exactly that, and is
    the test that keeps such compositions out of the "different algorithm" bin.
    """
    tot = wsum(sub_f)
    return wsum(sub_f & base_f) / tot if tot else 0.0


# ---------------------------------------------------------------------------
# editable-region extraction
# ---------------------------------------------------------------------------

def extract_region(variant_text: str, template_lines: list[str],
                   start: int, end: int) -> tuple[str, bool]:
    """Locate the editable region inside a *variant* of the template file.

    The region is declared by line numbers against the template, but a variant
    (a baseline edit, or an agent's submission) has different line numbers.  We
    align the two files with ``difflib.SequenceMatcher`` and carry the span
    [start-1, end) through the alignment.  That is robust both to the agent
    growing/shrinking the block and to the *task itself* having drifted since
    the run (e.g. the 2026-08-07 repairs edited fixed code outside the editable
    range in optimization-multi-objective) -- a naive prefix scan would stop at
    the first such line and swallow the rest of the file.

    Returns (region_src, anchors_intact), where anchors_intact is False if the
    fixed prefix/suffix does not align cleanly (agent clobbered protected code,
    or the shipped template moved under the run).
    """
    import difflib
    var = variant_text.splitlines()
    tpl = [ln.rstrip() for ln in template_lines]
    varn = [ln.rstrip() for ln in var]
    sm = difflib.SequenceMatcher(None, tpl, varn, autojunk=False)
    ops = sm.get_opcodes()

    lo_t, hi_t = start - 1, end

    def _map(i: int, upper: bool) -> int:
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= i < i2 or (i == i2 and tag == "equal"):
                if tag == "equal":
                    return j1 + (i - i1)
                return j2 if upper else j1
        return len(varn) if upper else 0

    lo_v, hi_v = _map(lo_t, False), _map(hi_t, True)
    lo_v = max(0, min(lo_v, len(var)))
    hi_v = max(lo_v, min(hi_v, len(var)))

    intact = all(
        tag == "equal"
        for tag, i1, i2, _, _ in ops
        if i2 <= lo_t or i1 >= hi_t
    )
    return "\n".join(var[lo_v:hi_v]), intact


def apply_ops(template_lines: list[str], ops: list[dict], target: str) -> list[str]:
    """Apply an edit-op list (baseline OPS) to the template, MLS-Bench semantics.

    ``target`` is the config-declared editable filename; an op matches if its
    ``file`` is that path or a suffix of it (the 2026-06 fix made task-body edit
    paths package-prefixed, so baseline files may carry either form).
    """
    lines = list(template_lines)
    def _matches(f: str) -> bool:
        f = (f or "").strip()
        return bool(f) and (f == target or target.endswith("/" + f) or f.endswith("/" + target))

    ordered = [o for o in ops if _matches(o.get("file", ""))]
    # apply bottom-to-top so earlier line numbers stay valid
    ordered.sort(key=lambda o: -(o.get("start_line") or o.get("after_line") or 0))
    for op in ordered:
        kind = op.get("op")
        content = op.get("content", "")
        clines = content.splitlines()
        if kind == "create":
            lines = clines
        elif kind == "replace":
            a, b = int(op["start_line"]), int(op["end_line"])
            lines = lines[: a - 1] + clines + lines[b:]
        elif kind == "insert":
            a = int(op["after_line"])
            lines = lines[:a] + clines + lines[a:]
    return lines


def load_ops_module(path: Path) -> list[dict]:
    """Exec an edit-op file in isolation and return its OPS (stdout suppressed)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_ops_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        spec.loader.exec_module(mod)
    return list(getattr(mod, "OPS", []))


# ---------------------------------------------------------------------------
# task reference bundle
# ---------------------------------------------------------------------------

_ALIAS_STOP = {
    "baseline", "baselines", "for", "the", "a", "an", "of", "and", "with", "in",
    "on", "to", "replaces", "reference", "paper", "uses", "task", "edit", "ops",
}


def _alias_tokens(key: str, docstring: str) -> set[str]:
    toks = {t for t in re.split(r"[_\-\s]+", key.lower()) if len(t) > 2}
    toks.add(key.lower().replace("_", ""))
    toks.add(key.lower().replace("_", " "))
    first = (docstring or "").strip().splitlines()[0] if docstring else ""
    # "BADGE baseline for ml-active-learning." -> BADGE
    head = re.split(r"\bbaseline\b", first, flags=re.I)[0]
    for m in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", head):
        if m.lower() not in _ALIAS_STOP:
            toks.add(m.lower())
    # ALLCAPS acronyms anywhere in the docstring
    for m in re.findall(r"\b[A-Z][A-Z0-9+]{2,}\b", docstring or ""):
        toks.add(m.lower())
    return {t for t in toks if t and t not in _ALIAS_STOP}


@dataclass
class BaselineRef:
    name: str
    canon: Canon
    aliases: set[str]
    disc: set[str] = field(default_factory=set)
    score: float | None = None


@dataclass
class TaskRef:
    name: str
    filename: str
    edit_start: int
    edit_end: int
    prefix: list[str]
    suffix: list[str]
    template_lines: list[str]
    template: Canon
    baselines: dict[str, BaselineRef]
    generic: set[str]
    required_symbols: set[str]
    best_baseline_score: float | None = None
    best_baseline_name: str | None = None
    template_baseline: str | None = None   # the stub already IS this named baseline
    stub_beats_best: bool | None = None

    def disc(self, feats: Iterable[str]) -> set[str]:
        """Discriminative features: drop anything the template already uses."""
        return {f for f in feats if f not in self.generic}


def load_task_ref(task: str, mls_root: Path) -> TaskRef:
    td = mls_root / "tasks" / task
    cfg = json.loads((td / "config.json").read_text())
    editfiles = [f for f in cfg.get("files", []) if f.get("edit")]
    if not editfiles:
        raise ValueError(f"{task}: no editable file in config.json")
    ef = editfiles[0]
    fn = ef["filename"]
    rng = ef["edit"][0]
    start, end = int(rng["start"]), int(rng["end"])

    tpl_path = td / "edits" / "custom_template.py"
    tpl_lines = tpl_path.read_text().splitlines()
    prefix = tpl_lines[: start - 1]
    suffix = tpl_lines[end:]
    tpl_region = "\n".join(tpl_lines[start - 1: end])
    tpl_canon = canonicalize(tpl_region)

    baselines: dict[str, BaselineRef] = {}
    for name, bc in (cfg.get("baselines") or {}).items():
        rel = bc.get("edit_ops")
        if not rel:
            continue
        p = td / rel
        if not p.exists():
            continue
        try:
            ops = load_ops_module(p)
        except Exception:
            continue
        blines = apply_ops(tpl_lines, ops, fn)
        region, _ = extract_region("\n".join(blines), tpl_lines, start, end)
        doc = ""
        try:
            doc = ast.get_docstring(ast.parse(p.read_text())) or ""
        except SyntaxError:
            pass
        baselines[name] = BaselineRef(name=name, canon=canonicalize(region),
                                      aliases=_alias_tokens(name, doc))

    generic = set(tpl_canon.features)
    for b in baselines.values():
        b.disc = {f for f in b.canon.features if f not in generic}

    ref = TaskRef(
        name=task, filename=fn, edit_start=start, edit_end=end,
        prefix=prefix, suffix=suffix, template_lines=tpl_lines,
        template=tpl_canon, baselines=baselines,
        generic=generic, required_symbols=set(tpl_canon.symbols),
    )
    # Does the shipped stub already implement one of the named baselines?  On
    # several MLS tasks it does (the template's default IS baseline:random /
    # :mean_impute / :greedy / ...), which is why "submit the stub untouched"
    # is a legitimate, sometimes competitive, submission.
    for n, b in baselines.items():
        if tpl_canon.parsed and b.canon.parsed and (
            tpl_canon.ast_alpha == b.canon.ast_alpha or tpl_canon.skeleton == b.canon.skeleton
        ):
            ref.template_baseline = n
            break
    _attach_baseline_scores(ref, mls_root)
    return ref


def _attach_baseline_scores(ref: TaskRef, mls_root: Path) -> None:
    """Score every ``baseline:*`` leaderboard row through the task's score_spec."""
    try:
        sys.path.insert(0, str(mls_root / "src"))
        from mlsbench.scoring.anchors import BaselineAnchors
        from mlsbench.scoring.evaluate import (
            _load_leaderboard_records, load_expanded_spec, score_record_details,
        )
    except Exception:
        return
    td = mls_root / "tasks" / ref.name
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            anchors = BaselineAnchors(td)
            spec = load_expanded_spec(td, anchors)
            if spec is None:
                return
            records = _load_leaderboard_records(td / "leaderboard.csv")
    except Exception:
        return
    best, best_name = None, None
    for name in ref.baselines:
        rows = [r for r in records if str(r.get("model", "")) in (f"baseline:{name}", name)]
        if not rows:
            continue
        vals = []
        for r in rows:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    s, _, ok = score_record_details(spec, r, anchors)
                if ok and s is not None and not math.isnan(s):
                    vals.append(float(s))
            except Exception:
                continue
        if not vals:
            continue
        sc = max(vals)
        ref.baselines[name].score = sc
        if best is None or sc > best:
            best, best_name = sc, name
    ref.best_baseline_score, ref.best_baseline_name = best, best_name


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    level: str
    reason: str
    family: str | None = None
    family_kind: str | None = None
    sim_top: float = 0.0
    sim_second: float = 0.0
    sim_all: dict[str, float] = field(default_factory=dict)
    anchors_intact: bool = True
    missing_symbols: list[str] = field(default_factory=list)


def _name_evidence(canon: Canon, bl: BaselineRef) -> bool:
    text = canon.prose
    for a in bl.aliases:
        if len(a) < 3:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", text):
            return True
    return False


def classify(sub: Canon, ref: TaskRef, anchors_intact: bool,
             score: float | None) -> Verdict:
    if not sub.parsed:
        return Verdict("BROKEN", "editable region does not parse",
                       anchors_intact=anchors_intact)
    missing = sorted(ref.required_symbols - sub.symbols)
    if missing:
        return Verdict("BROKEN", f"missing contract symbol(s): {', '.join(missing)}",
                       anchors_intact=anchors_intact, missing_symbols=missing)

    if sub.src.strip() == ref.template.src.strip():
        return Verdict("L0", "byte-identical to template", anchors_intact=anchors_intact)
    if sub.ast_norm == ref.template.ast_norm:
        return Verdict("L1", "AST-identical to template (docstring/comment/format only)",
                       anchors_intact=anchors_intact)
    if sub.ast_alpha == ref.template.ast_alpha:
        return Verdict("L1", "AST-identical to template up to local renaming",
                       anchors_intact=anchors_intact)
    if sub.skeleton == ref.template.skeleton:
        return Verdict("L2", "template skeleton with different constants",
                       anchors_intact=anchors_intact)

    dsub = ref.disc(sub.features)
    sims = {n: weighted_jaccard(dsub, b.disc) for n, b in ref.baselines.items()}
    ordered = sorted(sims.items(), key=lambda kv: -kv[1])
    top = ordered[0] if ordered else ("", 0.0)
    second = ordered[1][1] if len(ordered) > 1 else 0.0

    for n, b in ref.baselines.items():
        if sub.ast_alpha == b.canon.ast_alpha:
            return Verdict("L3", f"verbatim baseline:{n}", n, "replica",
                           top[1], second, sims, anchors_intact)
    for n, b in ref.baselines.items():
        if sub.skeleton == b.canon.skeleton:
            return Verdict("L3", f"baseline:{n} with retuned constants", n, "retuned",
                           top[1], second, sims, anchors_intact)
    if top[1] >= TAU_HIGH:
        return Verdict("L3", f"structurally matches baseline:{top[0]} (sim={top[1]:.2f})",
                       top[0], "structural", top[1], second, sims, anchors_intact)

    # Thin wrapper / composition of standard parts: the submission introduces no
    # method vocabulary beyond one baseline's.  Requires a non-trivial submission
    # so a two-line edit cannot match by having almost no features at all.
    if wsum(dsub) >= MIN_DISC_WEIGHT:
        subs = sorted(((n, subsumption(dsub, b.disc)) for n, b in ref.baselines.items()),
                      key=lambda kv: (-kv[1], -sims.get(kv[0], 0.0)))
        if subs and subs[0][1] >= TAU_SUB:
            n, s = subs[0]
            return Verdict("L3", f"adds nothing beyond baseline:{n} (subsumption={s:.2f})",
                           n, "subsumed", top[1], second, sims, anchors_intact)

    for n, s in ordered:
        if s >= TAU_LOW and _name_evidence(sub, ref.baselines[n]):
            return Verdict("L3", f"names and partially matches baseline:{n} (sim={s:.2f})",
                           n, "declared", top[1], second, sims, anchors_intact)

    bb = ref.best_baseline_score
    if score is not None and bb is not None and score > bb:
        return Verdict("L5", f"novel; score {score:.4f} > best baseline:{ref.best_baseline_name} {bb:.4f}",
                       None, None, top[1], second, sims, anchors_intact)
    return Verdict("L4", f"no baseline family match (max sim={top[1]:.2f} to {top[0]})",
                   None, None, top[1], second, sims, anchors_intact)


# ---------------------------------------------------------------------------
# episode discovery
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"vendor/workspace/([^/\s]+)/([^/\s]+)")


@dataclass
class Episode:
    run: str
    model: str
    task: str
    workspace: str | None
    score: float | None
    status: str
    verdict: Verdict | None = None
    stub_score: float | None = None


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def find_workspace(log_path: Path, task: str) -> str | None:
    try:
        with open(log_path, "r", errors="replace") as fh:
            for _ in range(400):
                line = fh.readline()
                if not line:
                    break
                m = _WS_RE.search(_strip_ansi(line))
                if m and m.group(1) == task:
                    return m.group(2)
    except OSError:
        return None
    return None


def scan_run(outdir: Path, mls_root: Path, refs: dict[str, TaskRef]) -> list[Episode]:
    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    model = summary.get("model", outdir.name)
    scores = {t.get("task"): t for t in summary.get("tasks", [])}
    eps: list[Episode] = []
    logs = sorted((outdir / "task_logs").glob("*.log")) if (outdir / "task_logs").is_dir() else []
    for log in logs:
        task = log.stem
        if task not in refs:
            try:
                refs[task] = load_task_ref(task, mls_root)
            except Exception:
                continue
        ref = refs[task]
        wsid = find_workspace(log, task)
        info = scores.get(task, {})
        score = info.get("score")
        status = info.get("status", "unknown")
        ep = Episode(run=outdir.name, model=model, task=task, workspace=wsid,
                     score=(float(score) if isinstance(score, (int, float)) else None),
                     status=status)
        if wsid is None:
            ep.verdict = Verdict("BROKEN", "no workspace recorded in log")
            eps.append(ep)
            continue
        sub_path = mls_root / "vendor" / "workspace" / task / wsid / ref.filename
        if not sub_path.exists():
            alt = list((mls_root / "vendor" / "workspace" / task / wsid).rglob(Path(ref.filename).name))
            sub_path = alt[0] if alt else sub_path
        if not sub_path.exists():
            ep.verdict = Verdict("BROKEN", "submitted file missing from workspace")
            eps.append(ep)
            continue
        region, intact = extract_region(sub_path.read_text(errors="replace"),
                                        ref.template_lines, ref.edit_start, ref.edit_end)
        ep.verdict = classify(canonicalize(region), ref, intact, ep.score)
        eps.append(ep)
    return eps


# ---------------------------------------------------------------------------
# stub scores + 2x2
# ---------------------------------------------------------------------------

# 2026-08-07: six always-zero tasks were repaired and the edit-tool contract
# changed on the same day.  Runs on either side of that date are not the same
# experiment -- on ml-anomaly-detection the untouched stub went from 0.000 to
# 0.502 -- so the stub reference is kept per cohort and pooling is warned about.
COHORT_SPLIT = "20260807"
COHORTS = ("pre_20260807", "post_20260807")


def episode_cohort(e: Episode) -> str:
    m = re.search(r"(\d{8})_\d{6}", e.workspace or "")
    if not m:
        return COHORTS[1]
    return COHORTS[0] if m.group(1) < COHORT_SPLIT else COHORTS[1]


def derive_stub_scores(eps: list[Episode],
                       fallback: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Stub score per (cohort, task) = median score of episodes that submitted the stub.

    Computed from the corpus being scanned so the reference is measured under the
    same harness as the episodes it is compared against; the bundled table is only
    a fallback for tasks the scan happens not to cover.
    """
    byt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for e in eps:
        if e.verdict and e.verdict.level in ("L0", "L1") and e.score is not None:
            byt[episode_cohort(e)][e.task].append(e.score)
    out = {c: dict(fallback.get(c, {})) for c in COHORTS}
    for c, tasks in byt.items():
        for t, vals in tasks.items():
            out.setdefault(c, {})[t] = statistics.median(vals)
    # a cohort with no observation of its own inherits the other's
    for c, other in ((COHORTS[0], COHORTS[1]), (COHORTS[1], COHORTS[0])):
        for t, v in out.get(other, {}).items():
            out.setdefault(c, {}).setdefault(t, v)
    return out


def cohort_drift(stub: dict[str, dict[str, float]], eps: list[Episode],
                 tol: float = 1e-6) -> list[tuple[str, float, float]]:
    """Tasks whose stub score moved across the 2026-08-07 boundary, if both are present."""
    seen = {episode_cohort(e) for e in eps}
    if len(seen) < 2:
        return []
    a, b = stub.get(COHORTS[0], {}), stub.get(COHORTS[1], {})
    return sorted((t, a[t], b[t]) for t in set(a) & set(b) if abs(a[t] - b[t]) > tol)


def two_by_two(eps: list[Episode], tol: float = 1e-9) -> dict[str, int]:
    cells = Counter()
    for e in eps:
        if e.verdict is None:
            continue
        if e.verdict.level == "BROKEN":
            cells["broken"] += 1
            continue
        if e.score is None:
            cells["unscored"] += 1
            continue
        if e.stub_score is None:
            cells["no_stub_ref"] += 1
            continue
        innov = e.verdict.level in INNOVATIVE
        better = e.score > e.stub_score + tol
        worse = e.score < e.stub_score - tol
        if innov:
            cells["innovative_and_better" if better else
                  ("innovative_but_worse" if worse else "innovative_and_tied")] += 1
        else:
            cells["conservative_and_better" if better else
                  ("conservative_and_worse" if worse else "conservative_and_safe")] += 1
    return dict(cells)


def innovation_rate(eps: list[Episode]) -> tuple[int, int, float]:
    viable = [e for e in eps if e.verdict and e.verdict.level != "BROKEN"]
    n_inn = sum(1 for e in viable if e.verdict.level in INNOVATIVE)
    return n_inn, len(viable), (n_inn / len(viable) if viable else float("nan"))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(eps: list[Episode], detail: bool = False) -> str:
    out = io.StringIO()
    by_run: dict[str, list[Episode]] = defaultdict(list)
    for e in eps:
        by_run[e.run].append(e)

    out.write("=" * 108 + "\n")
    out.write("MLS-BENCH INNOVATION LEVELS (reference-grounded: template + named baselines)\n")
    out.write("=" * 108 + "\n")
    hdr = f"{'run':44s} " + " ".join(f"{l:>6s}" for l in LEVELS) + f" {'innov%':>7s} {'mean':>7s}"
    out.write(hdr + "\n" + "-" * len(hdr) + "\n")
    for run in sorted(by_run):
        rl = by_run[run]
        c = Counter(e.verdict.level for e in rl if e.verdict)
        n_inn, n_via, rate = innovation_rate(rl)
        scored = [e.score for e in rl if e.score is not None]
        mean = statistics.mean(scored) if scored else float("nan")
        out.write(f"{run:44s} " + " ".join(f"{c.get(l,0):6d}" for l in LEVELS)
                  + f" {100*rate:6.1f}% {mean:7.4f}\n")

    out.write("\n" + "=" * 108 + "\n")
    out.write("2x2  (innovation x score-vs-stub)\n")
    out.write("=" * 108 + "\n")
    keys = ["innovative_and_better", "innovative_and_tied", "innovative_but_worse",
            "conservative_and_better", "conservative_and_safe", "conservative_and_worse",
            "broken", "unscored", "no_stub_ref"]
    out.write("  innovative = L4/L5 (a different algorithm from every named baseline);\n"
              "  better/worse is against the STUB score for that task, i.e. what the\n"
              "  untouched template scores -- so 'conservative_and_safe' is the cell that\n"
              "  says 'submitted the stub, kept the stub's points'.\n")
    hdr2 = f"{'run':44s} " + " ".join(f"{k[:11]:>12s}" for k in keys)
    out.write(hdr2 + "\n" + "-" * len(hdr2) + "\n")
    for run in sorted(by_run):
        cells = two_by_two(by_run[run])
        out.write(f"{run:44s} " + " ".join(f"{cells.get(k,0):12d}" for k in keys) + "\n")

    out.write("\n" + "=" * 108 + "\n")
    out.write("PER-TASK (pooled over the runs scanned)\n")
    out.write("=" * 108 + "\n")
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in eps:
        by_task[e.task].append(e)
    hdr3 = (f"{'task':42s} {'n':>4s} " + " ".join(f"{l:>6s}" for l in LEVELS)
            + f" {'stub':>7s} {'bestBL':>7s} {'stub/BL':>8s}  stub is")
    out.write(hdr3 + "\n" + "-" * len(hdr3) + "\n")
    for t in sorted(by_task):
        tl = by_task[t]
        c = Counter(e.verdict.level for e in tl if e.verdict)
        stub = next((e.stub_score for e in tl if e.stub_score is not None), None)
        bb = _BEST_BL.get(t)
        ratio = (stub / bb) if (stub is not None and bb) else float("nan")
        note = _TPL_BASELINE.get(t)
        out.write(f"{t:42s} {len(tl):4d} " + " ".join(f"{c.get(l,0):6d}" for l in LEVELS)
                  + f" {stub if stub is not None else float('nan'):7.4f}"
                  + (f" {bb:7.4f}" if bb is not None else f" {'--':>7s}")
                  + f" {ratio:8.2f}  " + (f"baseline:{note}" if note else "") + "\n")
    out.write("\n  stub/BL is the stub score as a fraction of the best named baseline's score.\n"
              "  'stub is baseline:X' means the shipped template ALREADY implements a named\n"
              "  baseline, so submitting it untouched is a legitimate standard submission.\n")

    if detail:
        out.write("\n" + "=" * 108 + "\n")
        out.write("EPISODE DETAIL\n")
        out.write("=" * 108 + "\n")
        for e in sorted(eps, key=lambda x: (x.run, x.task)):
            v = e.verdict
            out.write(f"{e.run:40s} {e.task:38s} {v.level:6s} "
                      f"score={e.score if e.score is not None else float('nan'):.4f} "
                      f"stub={e.stub_score if e.stub_score is not None else float('nan'):.4f} "
                      f"| {v.reason}\n")
    return out.getvalue()


_BEST_BL: dict[str, float] = {}
_TPL_BASELINE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# calibration + self-test
# ---------------------------------------------------------------------------

def cmd_calibrate(mls_root: Path, tasks: list[str]) -> int:
    """Cross-baseline similarity: distinct algorithms must sit below TAU_HIGH."""
    cross, same, subx = [], [], []
    print(f"{'task':42s} {'n_bl':>5s} {'max cross-baseline sim':>24s}  pair")
    print("-" * 100)
    for t in tasks:
        try:
            ref = load_task_ref(t, mls_root)
        except Exception as exc:
            print(f"{t:42s} ERROR {exc}")
            continue
        names = list(ref.baselines)
        worst, worst_pair = 0.0, ("", "")
        for i, a in enumerate(names):
            same.append(weighted_jaccard(ref.baselines[a].disc, ref.baselines[a].disc))
            for b in names[i + 1:]:
                s = weighted_jaccard(ref.baselines[a].disc, ref.baselines[b].disc)
                cross.append(s)
                subx.append(subsumption(ref.baselines[a].disc, ref.baselines[b].disc))
                subx.append(subsumption(ref.baselines[b].disc, ref.baselines[a].disc))
                if s > worst:
                    worst, worst_pair = s, (a, b)
        print(f"{t:42s} {len(names):5d} {worst:24.3f}  {worst_pair[0]} vs {worst_pair[1]}")
    if subx:
        subx.sort()
        print("\ncross-baseline SUBSUMPTION (does A add nothing beyond B?) distribution:")
        print(f"  n={len(subx)} mean={statistics.mean(subx):.3f} median={statistics.median(subx):.3f}")
        for q in (0.5, 0.9, 0.95, 0.99, 1.0):
            print(f"  p{int(q*100):3d} = {subx[min(len(subx)-1, int(q*(len(subx)-1)))]:.3f}")
        print(f"TAU_SUB={TAU_SUB} -> false 'same family' on distinct baselines: "
              f"{sum(1 for s in subx if s >= TAU_SUB)}/{len(subx)}")
    if cross:
        cross.sort()
        print("\ncross-baseline (different algorithms, same task) similarity distribution:")
        print(f"  n={len(cross)} mean={statistics.mean(cross):.3f} median={statistics.median(cross):.3f}")
        for q in (0.5, 0.9, 0.95, 0.99, 1.0):
            print(f"  p{int(q*100):3d} = {cross[min(len(cross)-1, int(q*(len(cross)-1)))]:.3f}")
        print(f"\nTAU_HIGH={TAU_HIGH}  -> false 'same family' on distinct baselines: "
              f"{sum(1 for s in cross if s >= TAU_HIGH)}/{len(cross)}")
        print(f"TAU_LOW ={TAU_LOW}   -> pairs needing name evidence: "
              f"{sum(1 for s in cross if TAU_LOW <= s < TAU_HIGH)}/{len(cross)}")
    return 0


# The ahc039 failure mode, verbatim: the model announces the hard method, then
# ships the easy one.  A keyword metric scores the announcement.  This metric
# must score the code, so a stub carrying this sentence has to stay conservative.
AHC039_PROSE = ("I cannot implement simulated annealing reliably, so I'll use a "
                "bounding box heuristic. This is a novel adaptive Bayesian "
                "information-theoretic approach.")


def _probe_declared_not_delivered(ref: TaskRef) -> Canon:
    """Stub + an ambitious claim in a comment and a docstring.  Must stay L0/L1."""
    body = ref.template.src
    doc_injected = body.replace('"""', f'"""{AHC039_PROSE} ', 1) if '"""' in body else body
    return canonicalize(f"# {AHC039_PROSE}\n" + doc_injected)


def _probe_renamed(ref: TaskRef) -> Canon | None:
    """Stub with every local alpha-renamed.  Must be L1, never a new method."""
    if not ref.template.parsed:
        return None
    tree = ast.parse(ref.template.ast_norm)
    ren = {n: f"zz_{n}" for n in _collect_locals(tree)}
    if not ren:
        return None
    tree = _AlphaRenamer(set(ren)).visit(tree)
    ast.fix_missing_locations(tree)
    return canonicalize(ast.unparse(tree))


def _probe_retuned(ref: TaskRef) -> Canon | None:
    """Stub with one numeric constant changed.  Must be L2, never a new method."""
    if not ref.template.parsed:
        return None
    tree = ast.parse(ref.template.ast_norm)
    hit = []

    class _Bump(ast.NodeTransformer):
        def visit_Constant(self, node):
            if not hit and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                hit.append(True)
                return ast.Constant(value=node.value + 7)
            return node

    tree = _Bump().visit(tree)
    if not hit:
        return None
    ast.fix_missing_locations(tree)
    return canonicalize(ast.unparse(tree))


def cmd_self_test(mls_root: Path, tasks: list[str]) -> int:
    """Known-answer tests.

    Every input here has a level we know a priori, so a regression in the
    thresholds or the canonical forms shows up as a FAIL rather than as a
    plausible-looking number in a report:

      template                -> L0
      each named baseline     -> L3 assigned to itself
                                 (or L0/L1 when the stub already IS that baseline)
      stub + ambitious prose  -> L0/L1  (the ahc039 "declared but not delivered" case)
      stub, locals renamed    -> L1
      stub, one constant bumped -> L2
    """
    fails = 0
    probe_fail = Counter()
    for t in tasks:
        ref = load_task_ref(t, mls_root)
        v = classify(ref.template, ref, True, None)
        ok_tpl = v.level == "L0"
        fails += (not ok_tpl)
        bl_ok, bl_bad = 0, []
        for n, b in ref.baselines.items():
            vb = classify(b.canon, ref, True, None)
            if vb.level == "L3" and vb.family == n:
                bl_ok += 1
            elif n == ref.template_baseline and vb.level in ("L0", "L1"):
                # the stub already IS this baseline -> L0/L1 is the right answer
                bl_ok += 1
            else:
                bl_bad.append(f"{n}->{vb.level}/{vb.family}")
        fails += len(bl_bad)

        probes: list[tuple[str, Canon | None, set[str]]] = [
            ("declared-not-delivered", _probe_declared_not_delivered(ref), {"L0", "L1"}),
            ("renamed", _probe_renamed(ref), {"L1"}),
            ("retuned", _probe_retuned(ref), {"L2"}),
        ]
        pr_bad = []
        for name, canon, expect in probes:
            if canon is None:
                continue
            got = classify(canon, ref, True, None).level
            if got not in expect:
                pr_bad.append(f"{name}->{got}")
                probe_fail[name] += 1
        fails += len(pr_bad)

        status = "OK " if ok_tpl and not bl_bad and not pr_bad else "FAIL"
        note = f"  [stub == baseline:{ref.template_baseline}]" if ref.template_baseline else ""
        print(f"{status} {t:42s} template->{v.level:5s} baselines {bl_ok}/{len(ref.baselines)}"
              f" probes {len(probes)-len(pr_bad)}/{len(probes)}{note}"
              + (f"  BAD: {', '.join(bl_bad + pr_bad)}" if (bl_bad or pr_bad) else ""))
    if probe_fail:
        print(f"\nprobe failures by kind: {dict(probe_fail)}")
    print(f"\nself-test failures: {fails}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def default_tasks(mls_root: Path, outdirs: list[Path]) -> list[str]:
    for d in outdirs:
        tl = d / "task_logs"
        if tl.is_dir():
            return sorted(p.stem for p in tl.glob("*.log"))
    return sorted(p.name for p in (mls_root / "tasks").iterdir() if (p / "config.json").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdirs", nargs="*", type=Path,
                    help="MLS-Bench run directories (outputs/cc_mlsbench_cpu_*)")
    ap.add_argument("--mls-root", type=Path, default=MLS_ROOT_DEFAULT)
    ap.add_argument("--json", type=Path, help="write per-episode records here")
    ap.add_argument("--csv", type=Path, help="write per-episode table here")
    ap.add_argument("--stub-scores", type=Path, default=STUB_SCORE_TABLE,
                    help="fallback stub-score table (task -> score of the untouched template)")
    ap.add_argument("--write-stub-scores", action="store_true",
                    help="refresh the stub-score table from the scanned corpus")
    ap.add_argument("--detail", action="store_true", help="print per-episode verdicts")
    ap.add_argument("--calibrate", action="store_true",
                    help="report cross-baseline similarity, to justify TAU_HIGH/TAU_LOW")
    ap.add_argument("--self-test", action="store_true",
                    help="classify the references themselves (template must be L0, each baseline L3/itself)")
    args = ap.parse_args(argv)

    tasks = default_tasks(args.mls_root, args.outdirs)
    if args.calibrate:
        return cmd_calibrate(args.mls_root, tasks)
    if args.self_test:
        return cmd_self_test(args.mls_root, tasks)
    if not args.outdirs:
        ap.error("give at least one run directory (or use --calibrate / --self-test)")

    refs: dict[str, TaskRef] = {}
    eps: list[Episode] = []
    for d in args.outdirs:
        if not (d / "task_logs").is_dir():
            print(f"[skip] {d}: no task_logs/", file=sys.stderr)
            continue
        eps.extend(scan_run(d, args.mls_root, refs))

    for t, r in refs.items():
        if r.best_baseline_score is not None:
            _BEST_BL[t] = r.best_baseline_score
        if r.template_baseline:
            _TPL_BASELINE[t] = r.template_baseline

    fallback: dict[str, dict[str, float]] = {}
    if args.stub_scores and args.stub_scores.exists():
        fallback = json.loads(args.stub_scores.read_text()).get("stub_scores_by_cohort", {})
    stub = derive_stub_scores(eps, fallback)
    for e in eps:
        e.stub_score = stub.get(episode_cohort(e), {}).get(e.task)

    if args.write_stub_scores:
        args.stub_scores.write_text(json.dumps(
            {"_comment": "score of the untouched template per task, median over L0/L1 episodes, "
                         "split at the 2026-08-07 task repairs + edit-contract change",
             "stub_scores_by_cohort": {c: dict(sorted(stub.get(c, {}).items())) for c in COHORTS},
             "best_baseline_scores": {k: v for k, v in sorted(_BEST_BL.items())}},
            indent=2) + "\n")
        print(f"[wrote] {args.stub_scores}", file=sys.stderr)

    print(report(eps, detail=args.detail))
    drift = cohort_drift(stub, eps)
    if drift:
        print("!" * 108)
        print("WARNING: this scan mixes runs from before and after the 2026-08-07 task repairs")
        print("         and edit-contract change.  The stub reference is applied per cohort, but")
        print("         run-level means and level counts are NOT comparable across the boundary")
        print("         on these tasks, whose untouched-stub score moved:")
        for t, a, b in drift:
            print(f"           {t:44s} pre={a:.4f}  post={b:.4f}  ({b - a:+.4f})")
        print("!" * 108)

    if args.json:
        args.json.write_text(json.dumps([{
            "run": e.run, "model": e.model, "task": e.task, "workspace": e.workspace,
            "score": e.score, "stub_score": e.stub_score, "status": e.status,
            "level": e.verdict.level if e.verdict else None,
            "reason": e.verdict.reason if e.verdict else None,
            "family": e.verdict.family if e.verdict else None,
            "family_kind": e.verdict.family_kind if e.verdict else None,
            "sim_top": e.verdict.sim_top if e.verdict else None,
            "sim_all": e.verdict.sim_all if e.verdict else None,
            "anchors_intact": e.verdict.anchors_intact if e.verdict else None,
        } for e in eps], indent=1) + "\n")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "model", "task", "level", "family", "family_kind",
                        "sim_top", "score", "stub_score", "beats_stub", "reason"])
            for e in eps:
                v = e.verdict
                w.writerow([e.run, e.model, e.task, v.level if v else "", v.family if v else "",
                            v.family_kind if v else "", f"{v.sim_top:.3f}" if v else "",
                            e.score, e.stub_score,
                            (e.score > e.stub_score) if (e.score is not None and e.stub_score is not None) else "",
                            v.reason if v else ""])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

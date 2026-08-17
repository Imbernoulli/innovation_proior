"""REASONING verifier — built on the OFFICIAL LLM360/Reasoning360 reward code.

verify(generation_text, problem) -> {"passed": bool, "detail": str}

The official guru reward functions live in
  verl/utils/reward_score/{zebra_puzzle,puzzles_dataset,graph_dataset,arcagi,
                           tablereason}.py
and are vendored VERBATIM under ./official/.  This verifier dispatches by
domain exactly like the official verl/utils/reward_score/__init__.py:

  logic__zebra_puzzle   -> zebra_puzzle.compute_score
  logic__ordering_puzzle-> puzzles_dataset.compute_score   (method='strict')
  logic__graph          -> graph_dataset.compute_score
  logic__arcagi1 / barc -> arcagi.compute_score
  table__*              -> tablereason.compute_score

Each official compute_score returns {"score": s, "acc": s}; for rejection
sampling we require a FULLY-correct answer, i.e. score == 1.0 -> passed.

NOTE on stem_web: the official guru verifier for stem_web is an LLM judge
(stem_llm_judge).  It is therefore NOT included here (those rows are excluded
from the worklist at build time); a deterministic verifier cannot soundly grade
them.  If needed they can be graded with the served rollout model as judge.
"""
import os
import re
import sys
import json
import importlib
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = os.path.join(_HERE, "official")
# Put the official dir on sys.path so the vendored `verl` shim package resolves
# the verbatim `from verl.utils.reward_score...` imports inside tablereason /
# prime_math.  (The shadowing flat math.py was removed; the canonical copy lives
# at official/verl/utils/reward_score/math.py.)
if _OFFICIAL not in sys.path:
    sys.path.append(_OFFICIAL)


# lazy module cache
_MODS = {}


def _mod(name):
    """Load an official scorer module by import path (flat domain scorers) or via
    the verl shim (tablereason)."""
    if name in _MODS:
        return _MODS[name]
    if name == "tablereason":
        m = importlib.import_module("verl.utils.reward_score.tablereason")
    else:
        path = os.path.join(_OFFICIAL, name + ".py")
        spec = importlib.util.spec_from_file_location("_official_" + name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    _MODS[name] = m
    return m


def _score_of(ret):
    """Official compute_score returns {'score':s,'acc':s} or a float/bool."""
    if isinstance(ret, dict):
        return float(ret.get("score", 0.0))
    return float(ret)


def _table_score_fixed(gen, gt):
    """Table reasoning, fixing a CONFIRMED false-positive in the official
    tablereason._check_single_answer: it calls math_equal(..., tolerance=1e-3)
    with include_percentage=True (default), so a numeric answer that is the gold
    times/over 100 is accepted (e.g. \\boxed{10} matches gt 1000, \\boxed{1001}
    matches 1000 under 1e-3 rel tol on 1000).  We reuse the official extraction
    but compare with include_percentage=False and NO magnitude tolerance for
    integer-like gold.
    """
    tr = _mod("tablereason")
    mathmod = tr.math
    math_equal = tr.math_equal

    model_output = str(gen).lower()
    gtruth = str(gt).lower()
    solution_str = model_output.split("</think>")[-1]
    answer_str = mathmod.last_boxed_only_string(solution_str)
    if answer_str is not None:
        answer = tr.drop_latex_text(mathmod.remove_boxed(answer_str))
    else:
        answer = solution_str

    def check_one(ans, g):
        ans = ans.strip()
        g = g.strip()
        # numeric path
        try:
            na = ans.replace(",", "").replace("%", "/100").replace("$", "").replace(":", "/").replace("\\", "")
            na = float(eval(na, {"__builtins__": {}}))
            g_is_int = bool(re.fullmatch(r"[-+]?\d+(\.0+)?", g))
            gf = float(g)
            if g_is_int:
                # exact integer equality (no percentage, no tolerance)
                return abs(na - gf) < 1e-9
            # non-integer gold: tight relative tolerance, percentage OFF
            # (2026-08-17) the old call passed is_close=True, which math_equal does NOT accept ->
            # TypeError -> swallowed by the except below -> string is_equiv -> every non-integer
            # table gold was judged FAIL for the entire campaign. Correct kwargs:
            if math_equal(na, g, include_percentage=False, tolerance=1e-4):
                return True
            # ROUNDING-AWARE fallback (2026-08-17): gold carries ~5 sig decimals (0.20187) while a
            # correct answer is naturally rounded ("20.19%" = 0.2019, "-5.09%" = -0.0509). Accept iff the
            # gold rounds to the model's value at the model's OWN precision — i.e. |na-gf| <= half a unit
            # in the last place the model wrote. This is NOT a magnitude/percent slack: it can't turn 10
            # into 1000, and a model that writes more digits is held to them.
            m_ = re.search(r"[-+]?\d*\.(\d+)", ans.replace(",", ""))
            dec = len(m_.group(1)) if m_ else 0
            scale = 0.01 if "%" in ans else 1.0        # the decimals were written in percent units
            ulp_half = 0.5 * (10 ** (-dec)) * scale
            return abs(na - gf) <= ulp_half + 1e-12
        except Exception:
            return mathmod.is_equiv(ans, g)

    if "|" not in gtruth:
        ok = check_one(answer, gtruth)
    else:
        gparts = sorted(x.strip() for x in gtruth.split("|"))
        aparts = sorted(x.strip() for x in answer.split("|"))
        ok = len(gparts) == len(aparts) and all(check_one(a, g) for a, g in zip(aparts, gparts))
    return {"score": 1.0 if ok else 0.0, "acc": 1.0 if ok else 0.0}


def _dispatch(domain, gen, gt):
    if domain == "logic__zebra":
        return _mod("zebra_puzzle").compute_score(gen, gt)
    if domain == "logic__ordering":
        return _mod("puzzles_dataset").compute_score(gen, gt, method="strict")
    if domain == "logic__graph":
        return _mod("graph_dataset").compute_score(gen, gt)
    if domain in ("logic__arcagi1", "logic__barc"):
        return _mod("arcagi").compute_score(gen, gt)
    if domain in ("table__hitab", "table__multihier"):
        return _table_score_fixed(gen, gt)   # fixed on top of official tablereason
    raise KeyError(domain)


def verify(generation_text, problem):
    domain = problem.get("domain")
    spec = problem.get("reward_spec", {})
    gt = spec.get("ground_truth")
    if gt is None:
        return {"passed": False, "detail": "no ground_truth"}
    try:
        ret = _dispatch(domain, generation_text or "", gt)
        score = _score_of(ret)
    except KeyError:
        return {"passed": False, "detail": f"no deterministic verifier for domain {domain!r}"}
    except Exception as e:
        return {"passed": False, "detail": f"scorer error ({domain}): {e}"}
    passed = (score >= 1.0)
    return {"passed": bool(passed),
            "detail": f"{domain} official score={score} -> {'PASS' if passed else 'FAIL'}"}


if __name__ == "__main__":
    prob = json.load(open(sys.argv[1]))
    gen = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
    print(json.dumps(verify(gen, prob), indent=2))

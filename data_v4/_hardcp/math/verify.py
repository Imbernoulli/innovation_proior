"""MATH verifier (deterministic, no LLM judge).

verify(generation_text, problem) -> {"passed": bool, "detail": str}

problem must contain "expected_answer".

Equality engine
---------------
Uses HuggingFace math_verify the CORRECT way (this was the bug in the first
version, which called bare parse() on raw strings and silently dropped symbolic
factors so that gold "3" matched a wrong "3\\pi"):

  * both gold and prediction are wrapped in $...$ and parsed with an explicit
    [LatexExtractionConfig(), ExprExtractionConfig()] config
    (prediction additionally uses boxed_match_priority=0),
  * math_verify.verify(gold_parsed, pred_parsed) decides equivalence.

This rejects  3 vs 3\\pi,  2\\sqrt{34} vs 2,  475362144 vs 475362145,
4.74e-11 vs 4.74e+11  while accepting  1/2 vs 0.5,  \\frac12 vs 1/2,
(1,2) vs \\left(1,2\\right),  \\sqrt2 vs \\sqrt{2}.

A conservative sympy check is used only as a *fallback* when math_verify errors,
and there is NO magnitude-scaled numeric tolerance (the source of off-by-one
false positives in v1).  Answer extraction prefers the last \\boxed{...}.
"""
import re

try:
    from math_verify import (parse as _mv_parse, verify as _mv_verify,
                             LatexExtractionConfig, ExprExtractionConfig)
    _HAVE_MV = True
except Exception:
    _HAVE_MV = False

import sympy


# --------------------------------------------------------------------------- #
# answer extraction
# --------------------------------------------------------------------------- #
def _find_boxed(text):
    idxs = [m.start() for m in re.finditer(r"\\boxed", text)]
    if not idxs:
        return None
    i = text.find("{", idxs[-1])
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


_FINAL_RE = re.compile(
    r"(?:final answer is|the answer is|answer\s*[:=])\s*\$?([^\n$]+)",
    re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_answer(text):
    """Extract the model's final answer.  We ONLY trust an explicit answer
    marker (\\boxed{...} preferred, else a 'the answer is ...' phrase).  We do
    NOT fall back to 'the last number anywhere in the text' — an incidental
    trailing number must not be scored as the answer (a false-positive channel
    flagged in review).  The served model is instructed to emit \\boxed{}."""
    if not text:
        return None
    boxed = _find_boxed(text)
    if boxed is not None:
        return boxed.strip()
    tail = text[-800:]
    m = None
    for m in _FINAL_RE.finditer(tail):
        pass
    if m:
        cand = m.group(1).strip().rstrip(". ")
        if cand:
            return cand
    return None


# --------------------------------------------------------------------------- #
# normalisation helpers (for the exact-match fast path only)
# --------------------------------------------------------------------------- #
def _strip_wrappers(s):
    s = s.strip()
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    s = re.sub(r"^\$+|\$+$", "", s)
    s = s.strip()
    m = re.match(r"^\\text\{(.*)\}$", s)
    if m:
        s = m.group(1).strip()
    return s


def _norm_exact(s):
    s = _strip_wrappers(s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.lower()


# --------------------------------------------------------------------------- #
# equality engine
# --------------------------------------------------------------------------- #
def _mv_configs():
    return ([LatexExtractionConfig(), ExprExtractionConfig()],
            [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])


def _mv_eq(gold, pred):
    """math_verify equivalence, used the documented (correct) way."""
    if not _HAVE_MV:
        return None
    gcfg, pcfg = _mv_configs()
    try:
        g = _mv_parse("$" + gold + "$", extraction_config=gcfg,
                      extraction_mode="first_match")
        p = _mv_parse("$" + pred + "$", extraction_config=pcfg,
                      extraction_mode="first_match")
        if not g or not p:
            return None
        # gold must be the first argument
        return bool(_mv_verify(g, p))
    except Exception:
        return None


def _sympy_eq(gold, pred):
    """Conservative symbolic fallback (no magnitude-scaled tolerance)."""
    def to_expr(s):
        s = _strip_wrappers(s)
        for parser in ("latex", "plain"):
            try:
                if parser == "latex":
                    from sympy.parsing.latex import parse_latex
                    return parse_latex(s)
                from sympy.parsing.sympy_parser import (
                    parse_expr, standard_transformations,
                    implicit_multiplication_application)
                tf = standard_transformations + (implicit_multiplication_application,)
                return parse_expr(s.replace("^", "**"), transformations=tf)
            except Exception:
                continue
        return None

    ge, pe = to_expr(gold), to_expr(pred)
    if ge is None or pe is None:
        return None
    try:
        if sympy.simplify(ge - pe) == 0:
            return True
    except Exception:
        pass
    # numeric fallback — only trust it for plain finite reals, and use a tight
    # tolerance that does NOT blow up with magnitude (a magnitude-scaled tol was
    # the off-by-one false-positive source in v1).  We require both an absolute
    # and a small relative bound to be satisfied.
    try:
        gv = complex(ge.evalf())
        pv = complex(pe.evalf())
        if gv.imag == 0 and pv.imag == 0:
            gv, pv = gv.real, pv.real
            diff = abs(gv - pv)
            # accept only if the difference is negligible in BOTH senses:
            #   absolute < 1e-8  AND  relative < 1e-9
            return diff <= 1e-8 and diff <= 1e-9 * max(1e-12, abs(gv))
    except Exception:
        pass
    return False


def answers_equal(gold, pred):
    if pred is None:
        return False
    gold, pred = str(gold), str(pred)
    # fast exact path (safe: only accepts literally-identical normalised strings)
    if _norm_exact(gold) == _norm_exact(pred) and _norm_exact(gold) != "":
        return True
    r = _mv_eq(gold, pred)
    if r is not None:
        return r          # trust math_verify's verdict (both True and False)
    # only if math_verify could not parse/compare do we fall back
    return bool(_sympy_eq(gold, pred))


# --------------------------------------------------------------------------- #
# LLM-judge fallback (DeepSeek V4 Flash, thinking OFF) for gold answers that
# math_verify STRUCTURALLY cannot grade: multi-value ("x=0 and x=1"), relations
# ("f(n)=0"), intervals/inequalities ("2014<=x<2016"), or natural language
# ("No solutions."). Validated STRICT on an adversarial set (rejects wrong /
# incomplete answers) so it admits no false positives into the SFT data.
# Disable with HARDCP_MATH_JUDGE=0. Key: data_v4/_hardcp/.deepseek_key.
# --------------------------------------------------------------------------- #
import os as _os

_JUDGE_MODEL = "deepseek-v4-flash"
_JUDGE_URL = "https://api.deepseek.com/chat/completions"
_JUDGE_SYS = (
    "You are a strict math answer grader. Decide whether the CANDIDATE is mathematically equivalent to the "
    "REFERENCE for the given problem. Apply standard conventions: variables like n,k are integers unless stated; "
    "interval notation [a,b) equals a<=x<b (and (a,b], etc. likewise); algebraically equal forms count as equal "
    "(e.g. 1/2=0.5; sqrt2=\\sqrt{2}; (-1)^(3n)=(-1)^n for integer n). Ignore formatting, delimiters, and phrasing. "
    "BUT answer NO if the candidate gives a different value/solution set, is incomplete (missing some solutions), "
    "or is not clearly equivalent. Reply EXACTLY one word: YES or NO.")


def _judge_key():
    try:
        return open(_os.path.join(_os.path.dirname(__file__), "..", ".deepseek_key")).read().strip()
    except Exception:
        return None


def _is_ungradeable(gold):
    """gold forms math_verify cannot match -> route to the strict LLM judge."""
    e = _strip_wrappers(str(gold)).strip()
    if not e:
        return True
    low = e.lower()
    if low in ("none", "no solution", "no solutions", "no solutions."):
        return True
    for w in (" for all", " where ", " such that", " and ", " or ", " as a ",
              "no solution", "infinitely many", "does not", "cannot", " if ",
              " when ", "proof", "arbitrary", "any "):
        if w in low:
            return True
    if "\\text" in e:
        return True
    if any(s in e for s in ("<", ">", "\\le", "\\ge", "\\leq", "\\geq", "\\in")):
        return True
    if "=" in e:               # a relation "lhs = rhs", not a bare value
        return True
    return False


def _llm_judge(problem_text, gold, generation_text):
    """Strict flash judge verdict True/False, or None if unavailable/errored."""
    if _os.environ.get("HARDCP_MATH_JUDGE", "1") == "0":
        return None
    boxed = _find_boxed(generation_text or "")
    cand = (f"Extracted \\boxed answer: {boxed}\n" if boxed else "") + \
           "Model's conclusion (tail):\n" + (generation_text or "")[-1500:]
    user = (f"Problem:\n{problem_text}\n\nREFERENCE answer:\n{gold}\n\n"
            f"CANDIDATE answer:\n{cand}\n\nEquivalent? YES or NO.")
    # (2026-08-17, user decision: "判官用自己就行了") The judge is the LOCAL Qwen3.8 service, period.
    # DeepSeek is no longer consulted (its account hit 402 and it silently degraded to string compare
    # for weeks). Local judge: thinking ON, temp 0, last YES/NO in content; 7/7 adversarial matrix.
    # Set HARDCP_JUDGE=deepseek to force the old external path (needs a funded key).
    if _os.environ.get("HARDCP_JUDGE", "local") == "deepseek":
        key = _judge_key()
        if key:
            try:
                import requests
                body = {"model": _JUDGE_MODEL,
                        "messages": [{"role": "system", "content": _JUDGE_SYS},
                                     {"role": "user", "content": user}],
                        "max_tokens": 8, "temperature": 0, "thinking": {"type": "disabled"}}
                r = requests.post(_JUDGE_URL, headers={"Authorization": f"Bearer {key}"},
                                  json=body, timeout=90)
                txt = (r.json()["choices"][0]["message"]["content"] or "").strip().upper()
                if txt.startswith("Y"):
                    return True
                if txt.startswith("N"):
                    return False
            except Exception:
                pass
    return _local_judge(user)


def _local_judge(user_msg):
    try:
        import requests
    except Exception:
        return None
    urls = [u.strip() for u in _os.environ.get(
        "HARDCP_LOCAL_JUDGE_URLS",
        "http://127.0.0.1:30001,http://127.0.0.1:30002,http://127.0.0.1:30000").split(",") if u.strip()]
    # Thinking ON with a real budget: with thinking off + 8 tokens the local model answered YES to
    # ln6/3 vs (ln2+ln5)/3 (a false positive that would poison training data). Let it work it out,
    # then read the FINAL YES/NO from the answer content. Strictness > recall for a data gate.
    body = {"model": "Qwen3.8-27B",
            "messages": [{"role": "system", "content": _JUDGE_SYS +
                          " Work through the comparison carefully (simplify both sides, check every "
                          "case) before answering. Your final line must be exactly YES or NO."},
                         {"role": "user", "content": user_msg}],
            "max_tokens": 4000, "temperature": 0}
    for u in urls:
        try:
            r = requests.post(u.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
            msg = r.json()["choices"][0]["message"]
            txt = (msg.get("content") or "").strip().upper()
            # final verdict = last YES/NO token in the answer content (not the reasoning)
            import re as _re
            hits = _re.findall(r"\b(YES|NO)\b", txt)
            if hits:
                return hits[-1] == "YES"
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
def verify(generation_text, problem):
    gold = problem.get("expected_answer")
    if gold is None:
        return {"passed": False, "detail": "no expected_answer in problem"}
    # Gold that math_verify structurally cannot grade -> strict LLM judge (sees
    # the problem + full conclusion, not just an extracted token).
    if _is_ungradeable(gold):
        v = _llm_judge(problem.get("problem", ""), str(gold), generation_text)
        if v is not None:
            return {"passed": bool(v),
                    "detail": f"llm-judge gold={gold!r} -> {'MATCH' if v else 'MISMATCH'}"}
        # judge unavailable -> fall through to deterministic (conservative False)
    pred = extract_answer(generation_text)
    if pred is None:
        return {"passed": False, "detail": "no answer extracted from generation"}
    ok = answers_equal(str(gold), str(pred))
    return {"passed": bool(ok),
            "detail": f"pred={pred!r} gold={gold!r} -> {'MATCH' if ok else 'MISMATCH'}"}


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) == 3:
        prob = json.load(open(sys.argv[1]))
        gen = open(sys.argv[2]).read()
        print(json.dumps(verify(gen, prob), indent=2))

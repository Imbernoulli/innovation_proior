#!/usr/bin/env python3
"""Corpus scanner for the defect classes Codex found (independently confirmed):
  (L) closure/global LEAK: a Format-B evaluator hands the candidate an object whose
      __closure__ exposes unmetered callables / hidden arrays (the fsx_C_0183 signature).
  (O) closed / known-optimum: statement|gen|strong reveal a provably-optimal construction
      or the ground-truth (Hadamard/Paley/Laderman/"maximal"/"the true law"...), OR the
      strong solution caps at ratio~1.0 on most cases (=> essentially optimal, not open-ended).
Outputs reports/scan_defects.json with a per-problem verdict + the regenerate list.
"""
import json, os, re, subprocess, tempfile
from pathlib import Path

SYNTH = Path(__file__).resolve().parent.parent
PROB = SYNTH / "problems"

REDFLAG = re.compile(
    r"hadamard|paley|sylvester|laderman|maximal (?:binary )?determinant|"
    r"optimal construction|provably optimal|attains the maxim|known optimum|"
    r"exact optimum|the true law|ground[- ]truth (?:law|formula)|closed[- ]form optimum",
    re.IGNORECASE)


def read(p):
    try: return p.read_text(errors="replace")
    except Exception: return ""


def fn_name(evaluator_txt, sols):
    m = re.search(r'FN_NAME\s*=\s*["\'](\w+)', evaluator_txt)
    if m: return m.group(1)
    for s in sols:
        mm = re.search(r'^def (\w+)\(', read(s), re.M)
        if mm: return mm.group(1)
    return None


def leak_probe(pdir, evaluator, sols):
    """Run the evaluator with an adversary that reports whether any argument it
    receives exposes callables/ndarrays through __closure__ (unmetered-oracle leak)."""
    fn = fn_name(read(evaluator), sols)
    if not fn:
        return None
    adv = f'''
import sys, json
_f = []
def _insp(o, path):
    clo = getattr(o, "__closure__", None)
    if clo:
        for c in clo:
            try: v = c.cell_contents
            except Exception: continue
            tn = type(v).__name__
            if callable(v) or "ndarray" in tn:
                _f.append(path + ":closure_" + tn)
    # also large hidden arrays passed directly
    if type(o).__name__ == "ndarray" and getattr(o, "size", 0) > 4:
        _f.append(path + ":direct_ndarray")
def {fn}(*a, **k):
    for i, x in enumerate(a): _insp(x, "arg%d" % i)
    for kk, x in k.items(): _insp(x, kk)
    print("LEAKPROBE:" + json.dumps(_f), file=sys.stderr)
    return None
'''
    p = pdir / ".advprobe.py"
    try:
        p.write_text(adv)
        # invoke exactly like a normal candidate: from the problem dir, relative names
        r = subprocess.run(["python3", evaluator.name, ".advprobe.py"],
                           capture_output=True, text=True, timeout=180, cwd=str(pdir))
    except Exception:
        return None
    finally:
        try: p.unlink()
        except Exception: pass
    ms = re.findall(r"LEAKPROBE:(\[.*\])", r.stderr or "")
    if not ms:
        return []
    try:
        return json.loads(ms[-1])   # last (fully-accumulated) report
    except Exception:
        return []


def strong_caps(pdir):
    """Fraction of cases where the strong solution hits ratio ~1.0 (essentially optimal)."""
    vj = pdir / "validation.json"
    if not vj.exists():
        return None
    v = json.loads(read(vj))
    for label, info in v.get("solutions", {}).items():
        if info.get("tier") == "strong":
            arr = info.get("scores") or info.get("vector") or []
            if not arr:
                return None
            return sum(1 for x in arr if x >= 0.98) / len(arr)
    return None


def main():
    rows = []
    regen = []
    for d in sorted(PROB.glob("*/")):
        pid = d.name
        sols = sorted((d / "solutions").glob("*")) if (d / "solutions").exists() else []
        is_b = (d / "evaluator.py").exists()
        text = read(d / "statement.md") + read(d / "statement.txt")
        for g in ["gen.py", "gen.cpp"]:
            text += read(d / g)
        for s in sols:
            if s.stem == "strong":
                text += read(s)
        flags = []
        # (O) red-flag tokens
        rf = REDFLAG.findall(text)
        if rf:
            flags.append("REDFLAG:" + ",".join(sorted(set(x.lower() for x in rf))[:3]))
        # (O) strong caps at 1.0 on most cases
        sc = strong_caps(d)
        if sc is not None and sc >= 0.5:
            flags.append(f"STRONG_CAPS:{sc:.2f}")
        # (L) leak probe for Format B -- only a closure hiding a callable/array is a LEAK
        #     (a directly-passed ndarray is the intended instance input, not a leak)
        leak = None
        if is_b:
            leak = leak_probe(d, d / "evaluator.py", sols)
            sig = sorted(set(x for x in (leak or []) if "closure_" in x))
            if sig:
                flags.append("LEAK:" + ",".join(sig[:3]))
        verdict = "FLAG" if flags else "clean"
        rows.append({"id": pid, "format": "B" if is_b else "stdout",
                     "verdict": verdict, "flags": flags})
        if flags:
            regen.append(pid)
    out = {"total": len(rows), "flagged": len(regen), "regen": sorted(regen), "rows": rows}
    (SYNTH / "reports" / "scan_defects.json").write_text(json.dumps(out, indent=2))
    print(f"scanned {len(rows)} | flagged {len(regen)}")
    from collections import Counter
    c = Counter(f.split(":")[0] for r in rows for f in r["flags"])
    print("flag types:", dict(c))
    for r in rows:
        if r["flags"]:
            print(f"  {r['id']} [{r['format']}] {r['flags']}")


if __name__ == "__main__":
    main()

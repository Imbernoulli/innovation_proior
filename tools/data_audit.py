#!/usr/bin/env python3
"""Audit the SFT corpus' LANDING POINT, to ground the data-remediation diagnosis.

Reproduces the data-picture numbers in experiments/DATA_REMEDIATION_zh.md §1.1 directly from
methods/*/results/{train_answer,reasoning,context}.md. The thesis it quantifies: the data lands on
research narrative + paper-level Python libraries, NOT on executable competition deliverables -- so a
model trained on it learns the *register* of innovation, not the *discipline* of shipping a correct,
self-contained, test-passing solution (which is exactly what FrontierCS/ALE reward).

Usage:  python tools/data_audit.py            # audit methods/
        python tools/data_audit.py --json     # machine-readable
"""
import glob, re, json, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STDIN = re.compile(r'\b(std::cin|cin\s*>>|scanf|getline|sys\.stdin|input\(\)|readline\(\))')
CPP = re.compile(r'```(?:cpp|c\+\+|cc)\b', re.I)
PY = re.compile(r'```(?:python|py)\b', re.I)
CLASS = re.compile(r'\bclass\s+\w+')
# reasoning markers
LAND = re.compile(r'(read (?:from )?stdin|standard input|passes the tests?|within the time limit|'
                  r'single[- ]file|self-contained|i/o contract|output format)', re.I)
FALLBACK = re.compile(r'(fall back|fallback|simplest correct|keep it simple|revert to|stick with the '
                      r'simple|too risky|not worth the risk|ship the simple|retreat to|abandon (?:this|the))', re.I)
# context markers
COMP = re.compile(r'(stdin|standard input|print the|output the|test cases?|time limit|competitive|'
                  r'1\s*second)', re.I)


def rd(p):
    try:
        return open(p, encoding='utf-8').read()
    except OSError:
        return ''


def pct(n, d):
    return f"{n} ({100*n/max(d,1):.1f}%)"


def main():
    ta = sorted(glob.glob(os.path.join(ROOT, 'methods/*/results/train_answer.md')))
    rea = sorted(glob.glob(os.path.join(ROOT, 'methods/*/results/reasoning.md')))
    ctx = sorted(glob.glob(os.path.join(ROOT, 'methods/*/results/context.md')))

    a = {'n': len(ta), 'code': 0, 'stdin': 0, 'cpp': 0, 'py': 0, 'cls': 0}
    for p in ta:
        t = rd(p)
        a['code'] += '```' in t
        a['stdin'] += bool(STDIN.search(t))
        a['cpp'] += bool(CPP.search(t))
        a['py'] += bool(PY.search(t))
        a['cls'] += bool(CLASS.search(t))

    r = {'n': len(rea), 'land': 0, 'fb': 0}
    for p in rea:
        t = rd(p)
        r['land'] += bool(LAND.search(t))
        r['fb'] += bool(FALLBACK.search(t))

    c = {'n': len(ctx), 'comp': sum(bool(COMP.search(rd(p))) for p in ctx)}

    report = {'landing': a, 'reasoning': r, 'context': c}
    if '--json' in sys.argv:
        print(json.dumps(report, indent=2))
        return

    print(f"== LANDING POINT (train_answer.md, n={a['n']}) ==")
    print(f"  has code fence : {pct(a['code'], a['n'])}")
    print(f"  reads stdin    : {pct(a['stdin'], a['n'])}   <- competition deliverable signal")
    print(f"  C++ fence      : {pct(a['cpp'], a['n'])}     <- FCS judge only extracts ```cpp")
    print(f"  Python fence   : {pct(a['py'], a['n'])}")
    print(f"  defines class  : {pct(a['cls'], a['n'])}     <- paper-level library, not single-file")
    print(f"== REASONING (reasoning.md, n={r['n']}) ==")
    print(f"  landing/I-O/test markers : {pct(r['land'], r['n'])}   <- never targets shipping a runnable solve")
    print(f"  fall-back markers        : {pct(r['fb'], r['n'])}")
    print(f"== CONTEXT (context.md, n={c['n']}) ==")
    print(f"  competition-like prompts : {pct(c['comp'], c['n'])}")
    print("\nVerdict: the landing point is research narrative + paper-level Python libraries, not")
    print("executable competition deliverables. See experiments/DATA_REMEDIATION_zh.md for the fix.")


if __name__ == '__main__':
    main()

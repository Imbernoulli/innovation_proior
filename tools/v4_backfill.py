#!/usr/bin/env python3
"""data_v4 segmented backfill (COLLEAGUE_PROPOSALS_REVIEW_zh.md §2.5 / §7-1).

Background
----------
`23ff22f29` is the last tree before the 07-21 audit-edit pass
(`tools/audit_edit_reasoning_workflow.js`).  That pass was supposed to
*de-templatize* `data_v4/*/reasoning.md` ("修复=题目特异性重写开头与 pitfall,非删减")
but instead cut them ~61% (median 17.5k -> 6.8k chars), taking four genuinely
valuable segment families with it:

    edge     edge-case / corner enumeration      ("**Edge cases ...**")
    retrace  hand re-trace with concrete values  ("Re-trace `{4}`, `S = 6`: ...")
    oracle   independent-oracle / differential testing
    ship     the final ship decision             ("**Final solution**" / "that is what I ship")

Three families were removed *correctly* and must never come back:

    opener   the fixed "**Reading the problem and pinning the contract.**" opener
    recap    the "**Causal recap.**" tail restatement (duplicates the answer)
    tic      "deliberately" / "convinced myself" phrasing

This tool diffs OLD (`23ff22f29`) against NEW (working tree) at paragraph
granularity, classifies every OLD paragraph that no longer survives in NEW, and
re-inserts only the four backfill families -- verbatim, minus tic phrasing.

Hard rules encoded here
-----------------------
* Only text that exists verbatim in the baseline may be re-inserted.  No
  sentence is ever synthesized (`tools/make_commit_coda.py` is NOT involved).
* Code fences are never re-inserted and never modified: only prose blocks are
  candidates, and NEW blocks are copied through byte-for-byte.
* The opener and the causal recap are never re-inserted, under any class.
* Only `data_v4/<unit>/reasoning.md` is ever written.  Prompt/statement files
  (`context.md`, `train_answer.md`, `verify/`) are never touched.

Usage
-----
    tools/v4_backfill.py --dry-run                      # all units, counts + per-unit table
    tools/v4_backfill.py --dry-run --json out.json      # machine-readable dump
    tools/v4_backfill.py --dry-run --show UNIT          # per-paragraph decisions for one unit
    tools/v4_backfill.py --apply --top 20               # apply to the 20 largest deletions
    tools/v4_backfill.py --apply --units a,b,c
"""
import argparse, json, os, re, subprocess, sys
from collections import Counter, defaultdict

BASELINE = "23ff22f29"
ROOT = "data_v4"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- git helpers
def git_out(args):
    return subprocess.run(["git"] + args, cwd=REPO, capture_output=True, text=True,
                          check=True).stdout

def baseline_units():
    """units that have reasoning.md in BOTH the baseline tree and the worktree."""
    old = set()
    for p in git_out(["ls-tree", "-r", BASELINE, "--name-only"]).splitlines():
        m = re.fullmatch(rf"{ROOT}/([^/]+)/reasoning\.md", p)
        if m:
            old.add(m.group(1))
    new = {d for d in os.listdir(os.path.join(REPO, ROOT))
           if os.path.isfile(os.path.join(REPO, ROOT, d, "reasoning.md"))}
    return sorted(old & new), sorted(new - old), sorted(old - new)

def old_text(unit):
    return git_out(["show", f"{BASELINE}:{ROOT}/{unit}/reasoning.md"])

def new_path(unit):
    return os.path.join(REPO, ROOT, unit, "reasoning.md")

def new_text(unit):
    with open(new_path(unit), encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------------- segmenting
FENCE = re.compile(r"^\s*```")

def segment(text):
    """Split into blocks on blank lines, fence-aware.

    Returns [(kind, text)] with kind in {"text", "code"}.  A fenced code block is
    always its own block and its content is never inspected or altered.
    """
    blocks, buf, in_fence = [], [], False
    def flush(kind="text"):
        if buf:
            s = "\n".join(buf).strip("\n")
            if s.strip():
                blocks.append((kind, s))
            buf.clear()
    for line in text.split("\n"):
        if FENCE.match(line):
            if not in_fence:
                flush()
                in_fence = True
                buf.append(line)
            else:
                buf.append(line)
                in_fence = False
                flush("code")
            continue
        if in_fence:
            buf.append(line)
            continue
        if not line.strip():
            flush()
        else:
            buf.append(line)
    flush("code" if in_fence else "text")
    # a lone markdown heading (`## Final solver`) is a header for the paragraph
    # that follows it, not a segment of its own -- merge the two.
    merged = []
    i = 0
    while i < len(blocks):
        kind, txt = blocks[i]
        if (kind == "text" and re.match(r"^#{1,6}\s+\S", txt) and "\n" not in txt
                and i + 1 < len(blocks) and blocks[i + 1][0] == "text"):
            merged.append(("text", txt + "\n\n" + blocks[i + 1][1]))
            i += 2
            continue
        merged.append((kind, txt))
        i += 1
    return merged

# ---------------------------------------------------------------- similarity
CODESPAN = re.compile(r"`([^`]+)`")
NUM = re.compile(r"\d+(?:\.\d+)?(?:\^\d+)?(?:e\d+)?")

def norm_words(t):
    t = CODESPAN.sub(lambda m: " " + m.group(1) + " ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.split()

def ngrams(words, n=4):
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

def concretes(t):
    """Concrete-value fingerprint of a paragraph: inline code spans + numerals.

    This is the signal that survives a *paraphrase*: the 07-21 pass rewrote much
    of the prose, so word n-grams under-report survival, but a re-trace or an
    edge enumeration that was kept still names the same `S = 0`, `{2, 4}`, `10^6`.
    """
    out = set()
    for m in CODESPAN.finditer(t):
        s = re.sub(r"\s+", "", m.group(1)).lower()
        if s:
            out.add(s)
    for m in NUM.finditer(re.sub(r"`[^`]*`", " ", t)):
        out.add(m.group(0))
    return out

def coverage(old_block, doc_grams, doc_conc):
    og = ngrams(norm_words(old_block))
    oc = concretes(old_block)
    g_cov = len(og & doc_grams) / len(og) if og else 1.0
    c_cov = len(oc & doc_conc) / len(oc) if oc else None
    return g_cov, c_cov, len(oc)

# thresholds (transparent, tune here).  Two bands, because a paragraph carrying
# many concrete values is noisier: a 0.60 hit rate over 12+ values already means
# the check survived the rewrite, while 0.60 over 4 values can be coincidence.
G_PRESENT, G_PARTIAL = 0.45, 0.25
C_PRESENT, C_PARTIAL, C_MIN = 0.70, 0.45, 3
C_MANY, C_MANY_PRESENT, C_MANY_PARTIAL = 12, 0.60, 0.38

def status_of(g_cov, c_cov, n_conc):
    c = c_cov if c_cov is not None else -1.0
    if (g_cov >= G_PRESENT
            or (n_conc >= C_MANY and c >= C_MANY_PRESENT)
            or (n_conc >= C_MIN and c >= C_PRESENT)):
        return "PRESENT"
    if (g_cov >= G_PARTIAL
            or (n_conc >= C_MANY and c >= C_MANY_PARTIAL)
            or (n_conc >= C_MIN and c >= C_PARTIAL)):
        return "PARTIAL"
    return "MISSING"

# ---------------------------------------------------------------- classifier
HDR = re.compile(r"^\*\*(.{1,120}?)\*\*", re.S)
MDHDR = re.compile(r"^#{1,6}\s+(.{1,120}?)\s*$", re.M)

RE_OPENER = re.compile(r"^\s*\**Reading the (problem|objective|task|input|statement|contract)", re.I)
RE_RECAP = re.compile(r"^(causal recap|recap|the causal chain)", re.I)
RE_EDGE = re.compile(r"edge case|corner case|\bthe corners\b|edge and failure|boundary case", re.I)
RE_SHIP = re.compile(r"^(final solution|final solver|final program|final code|final answer|"
                     r"final implementation|what i ship|the final (solution|program|code))", re.I)
RE_SHIP_BODY = re.compile(r"what i ship\b|that is what i ship|i ship\b|i submit\b|"
                          r"this is the version i (ship|submit)", re.I)
RE_ORACLE = re.compile(r"\boracle\b|differential(ly)?[- ]test|brute[- ]force|brute oracle|"
                       r"stress[- ]?test|cross[- ]check|randomi[sz]ed test|random test|"
                       r"reference implementation|zero mismatches|independent (brute|check|"
                       r"implementation|verif)|self[- ]?verif", re.I)
RE_RETRACE = re.compile(r"re-?trace|hand[- ]trace|trace(s|d)? (it|the|through|by hand)|"
                        r"^a (second )?trace|walk it|sanity[- ]check|"
                        r"checking the recurrence|confirming the recurrence|"
                        r"by hand on the sample|re-?verif|numeric self-check|"
                        r"^(first|second|third) trace", re.I)
RE_VERIF = re.compile(r"verif|check|trace|test|oracle|counterexample|edge|overflow|sentinel|"
                      r"corner|prove|proof|mismatch|assert|invariant|complexit|"
                      r"time limit|memory", re.I)
# body-level hand re-trace: a paragraph that walks concrete values through the
# algorithm ("`dp[3]`: `c=1`->`dp[2]+1=3` ...") without necessarily saying "trace"
RE_CONC_DIGIT = re.compile(r"`([^`]*\d[^`]*)`")
RE_WALK = re.compile(r"\b(trace|walk|run|step|hand|expected|matches|agrees|verif\w*|"
                     r"check|correct|prints?|output)\b", re.I)

RE_TIC = re.compile(r"\bdeliberately\b|\bconvinced myself\b|\bconvince myself\b", re.I)

def classify(block):
    """-> (klass, family).  klass in {KEEP-OUT, BACKFILL, OTHER}."""
    m = HDR.match(block)
    if not m:
        m = MDHDR.match(block)
    hdr = m.group(1).strip() if m else ""
    body = block[m.end():] if m else block
    head80 = block[:100]

    if RE_OPENER.match(block.lstrip()) or RE_OPENER.search(head80):
        return "KEEP-OUT", "opener"
    if hdr and RE_RECAP.match(hdr):
        return "KEEP-OUT", "recap"
    if not hdr and re.match(r"^\s*causal recap", block, re.I):
        return "KEEP-OUT", "recap"

    if RE_EDGE.search(hdr) or (not hdr and RE_EDGE.search(block[:160])):
        return "BACKFILL", "edge"
    if hdr and RE_SHIP.match(hdr):
        return "BACKFILL", "ship"
    if RE_SHIP_BODY.search(block):
        return "BACKFILL", "ship"
    if RE_RETRACE.search(hdr) or (not hdr and RE_RETRACE.search(block[:200])):
        if len(concretes(block)) >= 3:
            return "BACKFILL", "retrace"
    if RE_ORACLE.search(hdr) or RE_ORACLE.search(block):
        return "BACKFILL", "oracle"
    if len(RE_CONC_DIGIT.findall(block)) >= 6 and RE_WALK.search(block):
        return "BACKFILL", "retrace"
    if RE_TIC.search(block) and not RE_VERIF.search(block):
        return "KEEP-OUT", "tic-only"
    return "OTHER", ""

# ---------------------------------------------------------------- tic stripping
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*`(\-])")

def strip_tics(text):
    """Remove the three banned phrasings, minimally.

    * sentences containing "convinced myself" / "convince myself" are dropped
      (their content is the tic itself: a self-congratulation, not a check);
    * "Causal recap"-style restatement sentences are dropped;
    * the adverb "deliberately" is deleted in place, keeping its sentence.
    Nothing is added.  Returns (text, [notes]).
    """
    notes = []
    lines = []
    for line in text.split("\n"):
        if not re.search(RE_TIC, line) and not re.match(r"^\s*\**causal recap\b", line, re.I):
            lines.append(line)                      # untouched, byte for byte
            continue
        keep = []
        for sent in SENT_SPLIT.split(line):
            if re.search(r"\bconvinced? myself\b", sent, re.I):
                notes.append("dropped sentence: convinced myself")
                continue
            if re.match(r"^\s*\**causal recap\b", sent, re.I):
                notes.append("dropped sentence: causal recap")
                continue
            keep.append(sent)
        p = " ".join(keep)
        if re.search(r"\bdeliberately\b", p, re.I):
            notes.append("removed adverb: deliberately")
            p = re.sub(r",\s*deliberately\s*,", ",", p, flags=re.I)
            p = re.sub(r"\s+deliberately\b", "", p, flags=re.I)
            p = re.sub(r"\bdeliberately\s+", "", p, flags=re.I)
            p = re.sub(r"\bdeliberately\b", "", p, flags=re.I)
            p = re.sub(r"[ \t]{2,}", " ", p)
            p = re.sub(r" +([,.;:])", r"\1", p)
            p = re.sub(r",\s*([.;:!?])", r"\1", p)     # "Edge cases, deliberately." -> "Edge cases."
            p = re.sub(r",\s*,", ",", p)
            p = re.sub(r"\(\s*\)", "", p)
            p = re.sub(r"\*\*\s*([.,])", r"\1**", p) if p.count("**") % 2 else p
        lines.append(p.rstrip())
    return "\n".join(lines).strip(), notes

def dangling_colon_fix(text, next_is_code):
    """A ship/trace paragraph that introduced a code fence we do not carry back
    ends in ':' -- turn that into '.'.  Punctuation only, no words added."""
    if next_is_code and text.rstrip().endswith(":"):
        return text.rstrip()[:-1] + ".", True
    return text, False

# ---------------------------------------------------------------- per unit
def analyse(unit):
    o_raw, n_raw = old_text(unit), new_text(unit)
    o_blocks, n_blocks = segment(o_raw), segment(n_raw)
    n_text_blocks = [b for b in n_blocks if b[0] == "text"]
    doc_grams = set()
    doc_conc = set()
    for _, b in n_blocks:
        doc_grams |= ngrams(norm_words(b))
        doc_conc |= concretes(b)
    per_block_grams = [ngrams(norm_words(b)) for _, b in n_blocks]

    recs = []
    mono = 0          # anchors must be non-decreasing: OLD order is NEW order
    for i, (kind, blk) in enumerate(o_blocks):
        if kind == "code":
            recs.append(dict(i=i, kind="code", status="SKIP-CODE", klass="OTHER",
                             family="code", chars=len(blk), text=blk))
            continue
        g, c, nc = coverage(blk, doc_grams, doc_conc)
        st = status_of(g, c, nc)
        klass, fam = classify(blk)
        # best-matching NEW block (for placement)
        og = ngrams(norm_words(blk))
        best, best_s = -1, 0.0
        for j, ng in enumerate(per_block_grams):
            if not og or j < mono:
                continue
            sim = len(og & ng) / len(og)
            if sim > best_s:
                best, best_s = j, sim
        if best >= 0 and best_s >= 0.15:
            mono = best
        recs.append(dict(i=i, kind="text", status=st, klass=klass, family=fam,
                         g_cov=round(g, 3), c_cov=(None if c is None else round(c, 3)),
                         n_conc=nc, anchor=best, anchor_sim=round(best_s, 3),
                         chars=len(blk), text=blk))
    return dict(unit=unit, old_chars=len(o_raw), new_chars=len(n_raw),
                old_blocks=o_blocks, new_blocks=n_blocks, recs=recs)

def plan(unit_info, include_partial=False):
    """Decide insertions.  -> (list of {after, text, family, notes}, projected text)"""
    recs = unit_info["recs"]
    n_blocks = unit_info["new_blocks"]
    o_blocks = unit_info["old_blocks"]
    ins = []
    # anchor map: old index -> new index, for blocks that survived
    anchor_at = {}
    last_anchor = None
    for r in recs:
        if r["kind"] == "code":
            continue
        if r["status"] == "PRESENT" or (r["status"] == "PARTIAL" and r["anchor_sim"] >= 0.15):
            if r["anchor"] >= 0:
                anchor_at[r["i"]] = r["anchor"]
    wanted = {"MISSING"} | ({"PARTIAL"} if include_partial else set())
    for r in recs:
        if r["kind"] == "code" or r["klass"] != "BACKFILL" or r["status"] not in wanted:
            continue
        txt, notes = strip_tics(r["text"])
        if RE_OPENER.search(txt[:100]) or re.match(r"^\s*\**causal recap", txt, re.I):
            continue                      # belt and braces: never the opener/recap
        nxt = o_blocks[r["i"] + 1][0] if r["i"] + 1 < len(o_blocks) else None
        txt, fixed = dangling_colon_fix(txt, nxt == "code")
        if fixed:
            notes.append("dangling ':' -> '.' (code fence not carried back)")
        # a paragraph that points deictically at a code fence we are not carrying
        # back ("the solver follows", "the code below") cannot be made coherent
        # without writing a new sentence -- so it is dropped, not patched.
        if nxt == "code" and re.search(r"\bfollows\b|\bbelow\b|\bhere it is\b|"
                                       r"\bas follows\b|\bthis is the (code|file)\b",
                                       txt, re.I):
            continue
        if len(txt) < 60:
            continue                      # nothing of substance left after stripping
        # nearest surviving anchor before this paragraph
        pos = None
        for j in range(r["i"] - 1, -1, -1):
            if j in anchor_at:
                pos = anchor_at[j]
                break
        if pos is None:
            for j in range(r["i"] + 1, len(o_blocks)):
                if j in anchor_at:
                    pos = max(0, anchor_at[j] - 1)
                    break
        if pos is None:
            pos = len(n_blocks) - 1
        if r["family"] == "ship":
            # the ship decision closes the trace by definition: if every later OLD
            # text block is code or the (never-restored) causal recap, append at end
            tail = [o for o in recs if o["i"] > r["i"] and o["kind"] == "text"
                    and o["family"] != "recap"]
            if not tail:
                pos = len(n_blocks) - 1
        ins.append(dict(after=pos, old_i=r["i"], text=txt, family=r["family"],
                        notes=notes, chars=len(txt)))
    ins.sort(key=lambda d: (d["after"], d["old_i"]))
    return ins

def render(unit_info, ins):
    by_pos = defaultdict(list)
    for d in ins:
        by_pos[d["after"]].append(d["text"])
    out = []
    for j, (_, blk) in enumerate(unit_info["new_blocks"]):
        out.append(blk)
        out.extend(by_pos.get(j, []))
    return "\n\n".join(out).rstrip() + "\n"

# ---------------------------------------------------------------- self-check density
RE_SELFCHECK = re.compile(
    r"\bwait\b|\bhmm\b|\bverif\w*|\bactually\b|double[- ]check|sanity[- ]check|"
    r"\brecheck\b|re-?check|let me (test|try|check|verify|trace|re-?run)|"
    r"\bre-?trace\b|counterexample|\bmismatch\w*|\bcross[- ]check", re.I)

def selfcheck_density(text):
    """markers per 1k tokens, tokens estimated as chars/4 (repo convention)."""
    toks = max(1, len(text) / 4)
    return len(RE_SELFCHECK.findall(text)) / (toks / 1000.0)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--units", help="comma-separated unit names")
    ap.add_argument("--top", type=int, help="N units with the largest deletions")
    ap.add_argument("--include-partial", action="store_true",
                    help="also backfill paragraphs only partially covered in NEW "
                         "(default: off, to avoid local duplication)")
    ap.add_argument("--json", help="write machine-readable dump here")
    ap.add_argument("--show", help="print per-paragraph decisions for one unit")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if not (args.dry_run or args.apply or args.show):
        ap.error("need --dry-run, --apply or --show")

    units, only_new, only_old = baseline_units()
    if args.limit:
        units = units[:args.limit]

    infos, plans = {}, {}
    for u in units:
        info = analyse(u)
        infos[u] = info
        plans[u] = plan(info, args.include_partial)

    if args.show:
        u = args.show
        info = infos[u]
        print(f"# {u}  OLD {info['old_chars']} -> NEW {info['new_chars']} chars")
        for r in info["recs"]:
            head = re.sub(r"\s+", " ", r["text"])[:90]
            print(f"[{r['i']:>2}] {r['status']:<9} {r['klass']:<9} {r['family']:<8} "
                  f"g={r.get('g_cov')} c={r.get('c_cov')} n={r.get('n_conc')} | {head}")
        print("\n-- plan --")
        for d in plans[u]:
            print(f"  after NEW#{d['after']} <- old#{d['old_i']} [{d['family']}] "
                  f"{d['chars']}ch notes={d['notes']}")
        return

    # aggregate
    cls_counter = Counter()
    fam_counter = Counter()
    status_counter = Counter()
    matrix = Counter()          # (family, status) for the four backfill families
    rows = []
    for u in units:
        info, ins = infos[u], plans[u]
        for r in info["recs"]:
            if r["kind"] == "code":
                continue
            status_counter[r["status"]] += 1
            if r["klass"] == "BACKFILL":
                matrix[(r["family"], r["status"])] += 1
            if r["status"] in ("MISSING",):
                cls_counter[r["klass"]] += 1
                if r["klass"] == "BACKFILL":
                    fam_counter[r["family"]] += 1
                elif r["klass"] == "KEEP-OUT":
                    fam_counter["keepout:" + r["family"]] += 1
        proj = render(info, ins)
        rows.append(dict(unit=u, old=info["old_chars"], new=info["new_chars"],
                         sc_old=selfcheck_density(open(os.devnull).read() or "") if False else selfcheck_density("".join(b for _, b in info["old_blocks"])),
                         proj=len(proj), n_ins=len(ins),
                         fams=Counter(d["family"] for d in ins),
                         add=sum(d["chars"] for d in ins),
                         sc_new=selfcheck_density(new_text(u)),
                         sc_proj=selfcheck_density(proj),
                         deleted=info["old_chars"] - info["new_chars"]))

    if args.apply:
        sel = []
        if args.units:
            sel = [u.strip() for u in args.units.split(",") if u.strip()]
        elif args.top:
            sel = [r["unit"] for r in sorted(rows, key=lambda r: -r["deleted"])[:args.top]]
        else:
            ap.error("--apply needs --units or --top")
        for u in sel:
            if not plans[u]:
                print(f"skip {u}: nothing to backfill")
                continue
            txt = render(infos[u], plans[u])
            with open(new_path(u), "w", encoding="utf-8") as f:
                f.write(txt)
            fams = Counter(d["family"] for d in plans[u])
            print(f"applied {u}: +{len(plans[u])} segments "
                  f"({', '.join(f'{k}x{v}' for k, v in sorted(fams.items()))}) "
                  f"{infos[u]['new_chars']} -> {len(txt)} chars")
        return

    # dry-run report to stdout
    print(f"units scanned: {len(units)} (baseline {BASELINE}; "
          f"new-only {len(only_new)}, baseline-only {len(only_old)})")
    print("OLD paragraph status:", dict(status_counter))
    print("MISSING by class:", dict(cls_counter))
    print("MISSING detail  :", dict(fam_counter))
    tot_ins = sum(r["n_ins"] for r in rows)
    print(f"backfill insertions planned: {tot_ins} across "
          f"{sum(1 for r in rows if r['n_ins'])} units")
    import statistics as st
    print("chars  median NEW %d -> projected %d" %
          (st.median([r["new"] for r in rows]), st.median([r["proj"] for r in rows])))
    print("tok(chars/4) median NEW %d -> projected %d ; max projected %d" %
          (st.median([r["new"] for r in rows]) / 4, st.median([r["proj"] for r in rows]) / 4,
           max(r["proj"] for r in rows) / 4))
    print("self-check markers /1k tok (broad regex): median NEW %.2f -> projected %.2f "
          "(OLD baseline ceiling %.2f)" %
          (st.median([r["sc_new"] for r in rows]), st.median([r["sc_proj"] for r in rows]),
           st.median([r["sc_old"] for r in rows])))
    print("\nfour backfill families, OLD paragraph fate:")
    for fam in ("edge", "retrace", "oracle", "ship"):
        tot = sum(v for (f, _), v in matrix.items() if f == fam)
        print("  %-8s total %4d | PRESENT %4d  PARTIAL %4d  MISSING %4d" %
              (fam, tot, matrix[(fam, "PRESENT")], matrix[(fam, "PARTIAL")],
               matrix[(fam, "MISSING")]))
    print("\ntop 25 units by planned insertions:")
    for r in sorted(rows, key=lambda r: (-r["n_ins"], -r["add"]))[:25]:
        print("  %-38s del=%6d ins=%2d (%s) %6d -> %6d" %
              (r["unit"], r["deleted"], r["n_ins"],
               ",".join(f"{k}x{v}" for k, v in sorted(r["fams"].items())),
               r["new"], r["proj"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(baseline=BASELINE, units=len(units),
                           status=dict(status_counter), cls=dict(cls_counter),
                           fam=dict(fam_counter),
                           rows=[dict(r, fams=dict(r["fams"])) for r in rows],
                           plans={u: [dict(d, text=d["text"]) for d in plans[u]]
                                  for u in units if plans[u]}), f, ensure_ascii=False)
        print("wrote", args.json)

if __name__ == "__main__":
    main()

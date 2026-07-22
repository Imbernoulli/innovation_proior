#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for fsx_B_1068
   (Synthetic-Lexicon Sonnet under a strict iambic-inversion budget with scarce rhyme classes).

Prints one final line: "... Ratio: <float in [0,1]>" and exits 0.
"""
import sys
import re
import math
from collections import defaultdict

TARGET = "01"           # canonical iambic foot: unstressed, stressed
N_LINES = 14
N_SLOTS = 5              # words per line (every word is exactly 2 syllables -> 5*2 = 10 syllables)
ENT_WEIGHT = 6.0
RHYME_WEIGHT = 2.0      # rewards spreading distinct words across the 14 rhyme-critical slots
                        # specifically -- isolates the reservation decision's payoff from
                        # whatever the free (non-rhyme) slots separately achieve
PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]  # ABAB CDCD EFEF GG (0-indexed lines)

INT_RE = re.compile(r"^-?\d+$")


def fail(reason):
    print(f"INFEASIBLE: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) < 2:
        raise RuntimeError("malformed instance")
    w = int(toks[0]); budget = int(toks[1])
    pos = 2
    lexicon = []
    for _ in range(w):
        st, c, letter = toks[pos], toks[pos + 1], toks[pos + 2]
        pos += 3
        lexicon.append((st, int(c), letter))
    return w, budget, lexicon


def parse_output(path, w):
    with open(path) as f:
        text = f.read()
    raw = text.splitlines()
    while raw and raw[-1].strip() == "":
        raw.pop()
    if len(raw) != N_LINES:
        fail(f"expected exactly {N_LINES} lines, got {len(raw)}")
    lines = []
    for li, ln in enumerate(raw):
        toks = ln.split()
        if len(toks) != N_SLOTS:
            fail(f"line {li+1}: expected {N_SLOTS} tokens, got {len(toks)}")
        row = []
        for t in toks:
            if not INT_RE.match(t):
                fail(f"line {li+1}: non-integer token {t!r}")
            v = int(t)
            if v < 0 or v >= w:
                fail(f"line {li+1}: word index {v} out of range [0,{w})")
            row.append(v)
        lines.append(row)
    return lines


def word_inversion_positions(stress, slot):
    """Global within-line syllable positions (subset of {2*slot, 2*slot+1}) where this word's
    stress mismatches the canonical iambic foot "01"."""
    out = []
    for b in range(2):
        if stress[b] != TARGET[b]:
            out.append(2 * slot + b)
    return out


def score_lines(lines, lexicon, budget, check_feasible=True):
    """Returns (feasible, reason_or_None, F). If check_feasible is False, assumes the caller's
    own construction is trivially valid (used only for the checker's internal baseline) but we
    still run the SAME feasibility logic for defense-in-depth."""
    # rhyme-pair feasibility
    for (i, j) in PAIRS:
        ci = lexicon[lines[i][4]][1]
        cj = lexicon[lines[j][4]][1]
        if ci != cj:
            return False, f"rhyme mismatch: line {i+1} class {ci} != line {j+1} class {cj}", 0.0

    # meter-inversion budget
    total_inv = 0
    pos_multiset = []
    for li in range(N_LINES):
        for s in range(N_SLOTS):
            idx = lines[li][s]
            st = lexicon[idx][0]
            ps = word_inversion_positions(st, s)
            total_inv += len(ps)
            pos_multiset.extend(ps)
    if total_inv > budget:
        return False, f"inversion budget exceeded: used {total_inv} > budget {budget}", 0.0

    # objective
    U = [lines[li][s] for li in range(N_LINES) for s in range(N_SLOTS)]
    distinct_word_count = len(set(U))

    rhyme_words = [lines[li][4] for li in range(N_LINES)]
    rhyme_distinct = len(set(rhyme_words))

    letter_of = [w[2] for w in lexicon]
    best_chain = 1
    cur_len = 1
    cur_letter = letter_of[U[0]]
    cur_set = {U[0]}
    for k in range(1, len(U)):
        idx = U[k]
        letter = letter_of[idx]
        if letter == cur_letter and idx not in cur_set:
            cur_set.add(idx)
            cur_len += 1
        else:
            cur_letter = letter
            cur_set = {idx}
            cur_len = 1
        best_chain = max(best_chain, cur_len)

    if pos_multiset:
        cnt = defaultdict(int)
        for p in pos_multiset:
            cnt[p] += 1
        total = len(pos_multiset)
        entropy = 0.0
        for v in cnt.values():
            p = v / total
            entropy -= p * math.log2(p)
    else:
        entropy = 0.0

    F = distinct_word_count + best_chain + ENT_WEIGHT * entropy + RHYME_WEIGHT * rhyme_distinct
    return True, None, F


def build_baseline(lexicon):
    """Checker's own trivial, always-feasible construction: find the single class with the most
    0-inversion ("01") words and cycle through ONLY that class's distinct good words for the
    whole poem. Rhyme is trivially satisfied (every word shares one class); meter is trivially
    satisfied (0 inversions); no cross-class insight is used."""
    good_by_class = defaultdict(list)
    for idx, (st, c, letter) in enumerate(lexicon):
        if st == "01":
            good_by_class[c].append(idx)
    if not good_by_class:
        pool = [0]  # defensive fallback; the generator always guarantees "01" words exist
    else:
        best_c = min(good_by_class.keys(), key=lambda c: (-len(good_by_class[c]), c))
        pool = sorted(good_by_class[best_c])
    total_slots = N_LINES * N_SLOTS
    seq = [pool[i % len(pool)] for i in range(total_slots)]
    lines = [seq[i * N_SLOTS:(i + 1) * N_SLOTS] for i in range(N_LINES)]
    return lines


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    w, budget, lexicon = read_instance(in_path)
    lines = parse_output(out_path, w)

    feasible, reason, F = score_lines(lines, lexicon, budget)
    if not feasible:
        fail(reason)

    baseline_lines = build_baseline(lexicon)
    b_feasible, b_reason, B = score_lines(baseline_lines, lexicon, budget)
    if not b_feasible:
        # should never happen given the generator's guarantees; keep the checker safe
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print(f"F={F:.4f} B={B:.4f} total_inv_budget={budget}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()

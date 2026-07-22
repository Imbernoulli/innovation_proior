# TIER: strong
"""Insight: the rhyme scheme is a bipartite matching over scarce classes that must be solved
BEFORE any line is written.

Phase 1 -- RESERVATION (before a single word is placed): assign each of the 7 rhyme-scheme
pairs to a rhyme class by simulating the whole poem's demand at once (load-balance pairs across
classes by remaining "good-word slack"), instead of letting whichever pair is encountered first
grab the locally-best class and starve the classes later pairs will need.

Phase 2: fill each pair's two rhyme-critical (5th) words from its RESERVED class's own distinct
good-word pool (cycling only when a class's pool is smaller than its assigned demand).

Phase 3: spend the leftover inversion budget deliberately across DIFFERENT metrical feet (not
just the rhyme foot everyone is forced to touch) to raise the entropy bonus.

Phase 4: fill the remaining free slots by chasing the longest available run of distinct,
same-initial-letter words across line boundaries, switching to whichever letter still has the
most unspent supply once a run dries up.
"""
import sys
from collections import defaultdict

PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
N_LINES = 14
N_SLOTS = 5


def main():
    data = sys.stdin.read().split()
    w = int(data[0]); budget = int(data[1])
    pos = 2
    lexicon = []
    for _ in range(w):
        st, c, letter = data[pos], data[pos + 1], data[pos + 2]
        pos += 3
        lexicon.append((st, int(c), letter))

    goods = defaultdict(list)
    bads = defaultdict(list)
    for idx, (st, c, letter) in enumerate(lexicon):
        (goods if st == "01" else bads)[c].append(idx)
    all_classes = sorted(set(c for _, c, _ in lexicon))

    # ---- Phase 1: reservation (bipartite load-balance of pairs -> classes) ----
    assigned_pairs = {c: 0 for c in all_classes}
    pair_class = {}
    for pidx, (i, j) in enumerate(PAIRS):
        c_star = max(all_classes,
                     key=lambda c: (len(goods[c]) - 2 * assigned_pairs[c], -c))
        assigned_pairs[c_star] += 1
        pair_class[pidx] = c_star

    # ---- Phase 2: rhyme-critical words from the reserved class's own pool ----
    lines = [[None] * N_SLOTS for _ in range(N_LINES)]
    spent = set()
    cursor = defaultdict(int)  # per-class cycling cursor into goods[c]
    for pidx, (i, j) in enumerate(PAIRS):
        c = pair_class[pidx]
        pool = goods[c] if goods[c] else (bads[c] if bads[c] else [0])
        for line_idx in (i, j):
            v = pool[cursor[c] % len(pool)]
            cursor[c] += 1
            lines[line_idx][4] = v
            spent.add(v)

    # ---- Phase 3: deliberately spend leftover budget across distinct metrical feet ----
    # slot s in {0,1,2,3} maps to global syllable positions {2s, 2s+1}; the rhyme foot (s=4)
    # already collects whatever cost Phase 2 incurred, so target the OTHER four feet here.
    remaining_budget = budget
    planned = {}  # slot index (0..3) -> word index, all placed into line 0
    for s in range(4):
        if remaining_budget < 1:
            break
        best = None  # (cost, idx)
        for idx, (st, c, letter) in enumerate(lexicon):
            if idx in spent or st == "01":
                continue
            cost = sum(1 for b in range(2) if st[b] != "01"[b])
            if cost <= remaining_budget and (best is None or cost > best[0] or
                                              (cost == best[0] and idx < best[1])):
                best = (cost, idx)
        if best is None:
            break
        cost, idx = best
        planned[s] = idx
        spent.add(idx)
        remaining_budget -= cost
    for s, idx in planned.items():
        lines[0][s] = idx

    # ---- Phase 4: free slots -- chase the longest same-letter run across line boundaries ----
    letters_present = sorted(set(l for _, _, l in lexicon))

    # Phase 4 must stay budget-neutral (all deliberate spending already happened, tracked, in
    # Phase 3) -- so it only ever draws 0-inversion ("01") words, never a costly fallback.
    def unspent01_count(letter):
        return sum(1 for idx, (st, c, l) in enumerate(lexicon)
                    if l == letter and st == "01" and idx not in spent)

    def pick_on_letter(letter):
        for idx, (st, c, l) in enumerate(lexicon):
            if l == letter and st == "01" and idx not in spent:
                return idx
        return None

    def choose_new_letter():
        best_l, best_c = None, -1
        for l in letters_present:
            cnt = unspent01_count(l)
            if cnt > best_c:
                best_c, best_l = cnt, l
        return best_l if best_c > 0 else None

    active_letter = None
    for li in range(N_LINES):
        for s in range(4):
            if lines[li][s] is not None:  # already placed by Phase 3 (line 0)
                active_letter = lexicon[lines[li][s]][2]
                continue
            if active_letter is not None:
                idx = pick_on_letter(active_letter)
            else:
                idx = None
            if idx is None:
                new_letter = choose_new_letter()
                idx = pick_on_letter(new_letter) if new_letter else None
                if idx is None:
                    idx = lines[li][4]  # fully exhausted lexicon: safe fallback, reuse own class word
                active_letter = lexicon[idx][2]
            else:
                active_letter = lexicon[idx][2]
            lines[li][s] = idx
            spent.add(idx)
        active_letter = lexicon[lines[li][4]][2]

    print("\n".join(" ".join(map(str, row)) for row in lines))


if __name__ == "__main__":
    main()

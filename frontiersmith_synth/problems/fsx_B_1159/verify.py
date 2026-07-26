#!/usr/bin/env python3
# Deterministic checker for ballast-buddy-freelist (format C, MINIMIZE cost).
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys
from collections import OrderedDict


class Infeasible(Exception):
    pass


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


# ---------------------------------------------------------------------------
# instance parsing
# ---------------------------------------------------------------------------
def parse_instance(path):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    heap = int(toks[p]); page = int(toks[p + 1]); tlb = int(toks[p + 2]); n = int(toks[p + 3])
    p += 4
    events = []
    for _ in range(n):
        typ = toks[p]; p += 1
        if typ == 'A':
            i = int(toks[p]); s = int(toks[p + 1]); p += 2
            events.append(('A', i, s))
        elif typ == 'F':
            i = int(toks[p]); p += 1
            events.append(('F', i))
        elif typ == 'T':
            i = int(toks[p]); off = int(toks[p + 1]); p += 2
            events.append(('T', i, off))
        else:
            fail("bad instance event type")
    return heap, page, tlb, events


# ---------------------------------------------------------------------------
# shared buddy + TLB engine. alloc_chooser(free_blocks, tlb_od, id, size)->addr
#                             free_chooser(free_blocks, tlb_od, id, addr, size)->flag(0/1)
# Both callables may raise Infeasible to abort. Deterministic (no set/dict-
# order dependence beyond insertion order, which is itself event-order driven).
# ---------------------------------------------------------------------------
def simulate(events, heap, page, tlb_size, alloc_chooser, free_chooser):
    free_blocks = {0: heap}     # addr -> size, unsplit free leaves only
    alloc_map = {}              # id -> (addr, size)
    tlb_od = OrderedDict()      # page -> True, LRU order (front=LRU, back=MRU)
    misses = splits = coalesces = 0

    def find_ancestor(addr, size):
        for a, s in free_blocks.items():
            if s >= size and a <= addr < a + s and (addr - a) % size == 0:
                return a, s
        return None

    for ev in events:
        if ev[0] == 'A':
            _, id_, size = ev
            if id_ in alloc_map:
                raise Infeasible("id %d already live" % id_)
            addr = alloc_chooser(free_blocks, tlb_od, id_, size)
            if not isinstance(addr, int) or addr < 0 or addr >= heap:
                raise Infeasible("addr out of range")
            anc = find_ancestor(addr, size)
            if anc is None:
                raise Infeasible("addr %d not a valid free block for size %d" % (addr, size))
            a, s = anc
            del free_blocks[a]
            while s > size:
                s //= 2
                left, right = a, a + s
                if addr < right:
                    keep, other = left, right
                else:
                    keep, other = right, left
                free_blocks[other] = s
                splits += 1
                a = keep
            if a != addr:
                raise Infeasible("addr misaligned inside ancestor")
            alloc_map[id_] = (addr, size)
        elif ev[0] == 'F':
            _, id_ = ev
            if id_ not in alloc_map:
                raise Infeasible("free of unknown/dead id %d" % id_)
            addr, size = alloc_map.pop(id_)
            flag = free_chooser(free_blocks, tlb_od, id_, addr, size)
            if flag not in (0, 1):
                raise Infeasible("bad coalesce flag")
            free_blocks[addr] = size
            if flag == 1:
                a, s = addr, size
                while s < heap:
                    buddy = a ^ s
                    if free_blocks.get(buddy) == s:
                        del free_blocks[a]
                        del free_blocks[buddy]
                        a = min(a, buddy)
                        s *= 2
                        free_blocks[a] = s
                        coalesces += 1
                    else:
                        break
        else:  # 'T'
            _, id_, offset = ev
            if id_ not in alloc_map:
                raise Infeasible("touch of unknown/dead id %d" % id_)
            addr, size = alloc_map[id_]
            if not (0 <= offset < size):
                raise Infeasible("touch offset out of range")
            pg = (addr + offset) // page
            if pg in tlb_od:
                tlb_od.move_to_end(pg)
            else:
                misses += 1
                if len(tlb_od) >= tlb_size:
                    tlb_od.popitem(last=False)
                tlb_od[pg] = True
    return misses, splits, coalesces


# ---------------------------------------------------------------------------
# participant decision source: reads "<TYPE> <id> <value>" triples from the
# flat output token stream, one per A/F event, in the SAME order as the input
# trace (T events need no participant line). id must match the expected
# event's id -- this rules out reordering / skipping decisions.
# ---------------------------------------------------------------------------
def make_participant_choosers(out_tokens):
    pos = [0]

    def next_triple(expect_type, expect_id):
        if pos[0] + 3 > len(out_tokens):
            raise Infeasible("output ran out of decisions")
        typ = out_tokens[pos[0]]
        if typ != expect_type:
            raise Infeasible("expected %s decision, got %r" % (expect_type, typ))
        try:
            i = int(out_tokens[pos[0] + 1])
        except Exception:
            raise Infeasible("non-integer id token")
        if i != expect_id:
            raise Infeasible("decision id %d != expected %d" % (i, expect_id))
        try:
            v = int(out_tokens[pos[0] + 2])
        except Exception:
            raise Infeasible("non-integer value token")
        pos[0] += 3
        return v

    def alloc_chooser(free_blocks, tlb_od, id_, size):
        return next_triple('A', id_)

    def free_chooser(free_blocks, tlb_od, id_, addr, size):
        return next_triple('F', id_)

    return alloc_chooser, free_chooser


# ---------------------------------------------------------------------------
# internal baseline B: a deliberately naive, feasible construction built by
# the checker itself from the instance only (no participant knowledge): a
# textbook allocator that is actively BLIND to locality -- whenever several
# free blocks of the right size exist, it AVOIDS whichever sector the crew
# roster currently remembers (the opposite of the innovation this problem is
# about) and it ALWAYS eager-coalesces on free. This is a weak reference, not
# a competitive strategy -- matching it earns only a small fraction of score.
# ---------------------------------------------------------------------------
def make_base_choosers(page):
    def base_alloc_chooser(free_blocks, tlb_od, id_, size):
        exact = [a for a, s in free_blocks.items() if s == size]
        if exact:
            cold = [a for a in exact if (a // page) not in tlb_od]
            if cold:
                return max(cold)
            rank = {pg: i for i, pg in enumerate(tlb_od.keys())}
            return min(exact, key=lambda a: rank[a // page])
        bigger = [(s, a) for a, s in free_blocks.items() if s > size]
        if not bigger:
            bigger = [(s, a) for a, s in free_blocks.items() if s >= size]
        bigger.sort()
        min_s = bigger[0][0]
        same = [a for s, a in bigger if s == min_s]
        return max(same)

    def base_free_chooser(free_blocks, tlb_od, id_, addr, size):
        return 1

    return base_alloc_chooser, base_free_chooser


def main():
    heap, page, tlb_size, events = parse_instance(sys.argv[1])

    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    alloc_chooser, free_chooser = make_participant_choosers(out_tokens)
    try:
        misses, splits, coalesces = simulate(events, heap, page, tlb_size, alloc_chooser, free_chooser)
    except Infeasible as e:
        fail(str(e))
    F = misses + 0.01 * (splits + coalesces)

    base_alloc_chooser, base_free_chooser = make_base_choosers(page)
    try:
        b_misses, b_splits, b_coalesces = simulate(events, heap, page, tlb_size,
                                                    base_alloc_chooser, base_free_chooser)
    except Infeasible as e:
        # the checker's own reference construction must always be feasible;
        # if it somehow is not, fail closed rather than crash (checker MUST exit 0).
        fail("internal baseline construction failed: %s" % e)
    B = b_misses + 0.01 * (b_splits + b_coalesces)
    if B <= 0:
        B = 1e-9

    # scale calibrated (see AGENT_BRIEF innovation addendum: "rescale the
    # baseline if strong saturates") so the ladder spreads trivial/greedy/
    # strong meaningfully instead of bunching near the floor; still leaves
    # strong far short of the cap.
    SCALE = 0.32
    sc = min(1000.0, 1000.0 * SCALE * B / max(1e-9, F))
    print("misses=%d splits=%d coalesces=%d F=%.2f baseMisses=%d baseSplits=%d "
          "baseCoalesces=%d B=%.2f Ratio: %.6f" %
          (misses, splits, coalesces, F, b_misses, b_splits, b_coalesces, B, sc / 1000.0))


if __name__ == "__main__":
    main()

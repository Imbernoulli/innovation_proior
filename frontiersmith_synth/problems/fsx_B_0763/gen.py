#!/usr/bin/env python3
"""gen.py <testId> -> prints one instance of "The Innkeeper's Ledger" to stdout.

Instance:
  line 1: n m s
  next n lines: r1 r2 f   (regular i's two favorite rooms, 1<=r1,r2<=m, r1!=r2; f = trace
                           frequency weight, positive int)

Deterministic: all randomness seeded from testId only.

Structure planted on purpose (>=3 of the 10 cases, in fact cases 3..10): a "hub" gadget
(see hub_gadget() below). Two rooms P and Q each get exactly 2 "anchor" regulars who
always seat cleanly (2 candidates, 2 cots), PLUS one "squeezed" regular S whose only two
candidate rooms are P and Q themselves (no private fallback). By the time S is written
into the ledger, both P and Q are already fully booked by their anchors, so a naive
walk-in-order seating (no eviction) sends S straight to the annex. A textbook cuckoo-style
insertion (evict-and-relocate on collision) rescues S -- but a COST-OBLIVIOUS eviction
always tries P before Q and always bumps whichever anchor its search meets FIRST,
regardless of that anchor's value. The generator makes the anchor it always bumps (A1)
the expensive one and leaves a cheap anchor (A2) at the very same room unevicted, so a
frequency-aware reformulation (which anchor is cheapest to relegate?) saves substantially
more than the structurally-fixed eviction ever finds.
"""
import random, sys


def hub_gadget(rnd, room_base):
    """A five-regular gadget over 6 rooms: P, Q, PA1, PA2, QD1, QD2 (= room_base..+5).

    A1, A2 both list P as choice-1 (their own private choice-2 rooms are PA1, PA2).
    D1, D2 both list Q as choice-1 (private choice-2 rooms QD1, QD2). S lists P as
    choice-1 and Q as choice-2 -- ITS OWN candidates are exactly the two rooms the
    anchors already fill, with no private escape.

    Ledger order is A1, A2, D1, D2, S (anchors booked before the squeezed regular even
    arrives -- an entirely ordinary "seat people as they check in" order). By the time S
    is processed, P and Q both already show 2/2 cots taken.
      - A naive walk-in-order seating (no eviction) then has nowhere to put S: annex,
        cost 10*fS, even though a cheap real cot was one relocation away.
      - A textbook "evict on collision" cuckoo insertion tries S's P-slots before its
        Q-slots and, on the first collision it meets, evicts A1 (who has a free private
        cot PA1 to retreat to) -- ALWAYS A1, regardless of A1's frequency, because the
        search order is fixed, not value-aware.
      - fA1 is built large and fA2 small: a min-cost reformulation instead compares what
        it actually costs to relegate each anchor and evicts the CHEAP one (A2), leaving
        the expensive A1 exactly where it was. That comparison -- who is cheapest to
        relegate, not who the search meets first -- is invisible to a fixed traversal
        order and is exactly the nonlocal, nonobvious payoff of spending the rescue on
        the right regular.
    """
    P, Q, PA1, PA2, QD1, QD2 = (room_base + i for i in range(6))
    fA1 = rnd.randint(300, 480)
    fA2 = rnd.randint(20, 80)
    fD1 = rnd.randint(50, 150)
    fD2 = rnd.randint(50, 150)
    fS = rnd.randint(150, 280)
    keys = [
        (P, PA1, fA1),
        (P, PA2, fA2),
        (Q, QD1, fD1),
        (Q, QD2, fD2),
        (P, Q, fS),
    ]
    return keys, room_base + 6


def make_case(test_id):
    rnd = random.Random(2027041 * test_id + 13)

    # difficulty ladder over test_id: small/plain -> large/adversarial.
    # "hubs" = number of hub_gadget copies (each is 5 keys / 6 rooms). Cases 3..10
    # (>= 3 of the 10) carry hubs, i.e. the planted trap.
    sizes = {
        1: dict(n_filler=3, hubs=1),
        2: dict(n_filler=4, hubs=2),
        3: dict(n_filler=3, hubs=3),
        4: dict(n_filler=4, hubs=5),
        5: dict(n_filler=5, hubs=7),
        6: dict(n_filler=6, hubs=10),
        7: dict(n_filler=8, hubs=14),
        8: dict(n_filler=10, hubs=19),
        9: dict(n_filler=12, hubs=25),
        10: dict(n_filler=15, hubs=32),
    }[test_id]

    # keys are assembled as BLOCKS (one hub = one block of 5, kept internally ordered;
    # one filler regular = a block of 1) and only the BLOCK order is shuffled. This
    # varies the ledger's macro layout across test cases without ever breaking a hub's
    # internal A1,A2,D1,D2,S ledger order (the property the trap depends on).
    blocks = []
    room_cursor = 1
    for _ in range(sizes["hubs"]):
        gk, room_cursor = hub_gadget(rnd, room_cursor)
        blocks.append(gk)

    # a pool of "plain" rooms for filler regulars, generous relative to filler count so
    # capacity is never remotely tight there
    plain_room_lo = room_cursor
    n_plain_rooms = max(4, (sizes["n_filler"] * 2) // 3 + 2)
    plain_rooms = list(range(plain_room_lo, plain_room_lo + n_plain_rooms))
    room_cursor += n_plain_rooms

    m = room_cursor - 1

    for _ in range(sizes["n_filler"]):
        r1, r2 = rnd.sample(plain_rooms, 2)
        tier = rnd.random()
        if tier < 0.7:
            f = rnd.randint(1, 4)          # low
        else:
            f = rnd.randint(10, 40)        # medium
        blocks.append([(r1, r2, f)])

    rnd.shuffle(blocks)
    keys = [k for block in blocks for k in block]
    n = len(keys)
    # annex sized generously enough that BOTH the naive walk-in-order baseline and the
    # cuckoo-insertion greedy always remain feasible (never truncated) -- the gap
    # between tiers comes purely from HOW WELL they use their annex/room budget, not
    # from one of them failing outright.
    s = max(3, sizes["hubs"] + n // 10)

    return n, m, s, keys


def main():
    test_id = int(sys.argv[1])
    n, m, s, keys = make_case(test_id)
    out = [f"{n} {m} {s}"]
    for (r1, r2, f) in keys:
        out.append(f"{r1} {r2} {f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

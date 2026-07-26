import sys, random

# ---------------------------------------------------------------------------
# ballast-buddy-freelist  (format C, MINIMIZE total pilot cost)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.
#   Deterministic in testId only.
#
# Theme: a harbor pilot assigns berths (a buddy-system quay of length HEAP,
# power-of-two berth sizes) to ships as they arrive/depart, while tugboats
# (the harbor crew) repeatedly visit moored ships. Visits only go smoothly if
# the crew's short-term "sector roster" (a TLB_SIZE-entry, exact-LRU cache of
# 4KB quay sectors) already lists the ship's sector; otherwise the crew must
# re-brief on that sector (a "miss", cost 1). Freed berths may be joined with
# their sibling immediately or left alone for now.
#
# Instance (stdout):
#   line 1:  HEAP PAGE TLB N
#   N lines, one event each, in order:
#     "A id size"    -- ship <id> arrives, requests a <size>-byte berth
#     "F id"         -- ship <id> departs (berth freed)
#     "T id offset"  -- tugboat visits ship <id> at byte offset <offset>
#                       (0 <= offset < that ship's size)
#
# ids are unique for the whole trace (never reused). sizes are powers of two
# drawn from a fixed small menu, each strictly smaller than PAGE so every
# berth lies inside exactly one sector (no berth straddles two sectors).
#
# Construction: build a large RESIDENT fleet, then run several RECYCLE rounds
# that free a batch of ships and immediately re-request the SAME multiset of
# sizes (creating genuine same-size free-block choices) with each new arrival
# worked hard right away -- this is where a placement policy's locality
# choice actually pays off or costs. Explicit trap bursts are woven into
# several recycle rounds.
# ---------------------------------------------------------------------------

HEAP = 1 << 20      # 1,048,576 byte quay
PAGE = 1 << 12       # 4096-byte sectors  (256 sectors total)
TLB = 16             # crew remembers only 16 of the 256 sectors (exact LRU)
SIZES = [64, 128, 256, 512, 1024]


def insert_burst(rng, events, alive, next_id, burst_idx):
    """4 same-size ships arrive back-to-back; only ONE (b2) is kept busy
    (warm sector); both b1 (cold) and b2 (warm) then depart, and a NEW ship
    of the identical size arrives immediately and is worked hard. Reusing
    b2's warm sector for the newcomer avoids re-briefing costs; the lowest-
    address / eager-coalesce reflex reaches for b1 instead (or forces a
    fresh cold sector via an unnecessary re-split), paying for it on every
    subsequent visit."""
    BURST_SIZES = [128, 256, 512]
    S = BURST_SIZES[burst_idx % len(BURST_SIZES)]
    ids = list(range(next_id, next_id + 4))
    for i in ids:
        events.append(('A', i, S))
        alive[i] = S
    next_id += 4
    cold_id, warm_id, extra1, extra2 = ids
    for _ in range(8):
        events.append(('T', warm_id, rng.randrange(S)))
    events.append(('F', cold_id)); del alive[cold_id]
    events.append(('F', warm_id)); del alive[warm_id]
    new_id = next_id; next_id += 1
    events.append(('A', new_id, S))
    alive[new_id] = S
    for _ in range(8):
        events.append(('T', new_id, rng.randrange(S)))
    for i in (extra1, extra2):
        events.append(('F', i)); del alive[i]
    return next_id


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    rng = random.Random(20260726 + 731 * t)

    R = 150 + 25 * t                 # resident fleet size: 175 .. 400
    rounds = 3 + t // 2              # recycle rounds: 3 .. 8
    n_bursts = t // 3                # 0,0,1,1,1,2,2,2,3,3 for t=1..10

    events = []
    alive = {}                       # id -> size
    next_id = 1

    # ---- phase 1: build the resident fleet (no touches yet: first use of
    # every one of these bytes is unavoidably cold under ANY policy) --------
    for _ in range(R):
        size = rng.choice(SIZES)
        events.append(('A', next_id, size))
        alive[next_id] = size
        next_id += 1

    # a light warm-up pass so the cache starts non-empty
    ids_now = list(alive.keys())
    rng.shuffle(ids_now)
    for i in ids_now[:min(len(ids_now), TLB * 3)]:
        events.append(('T', i, rng.randrange(alive[i])))

    # ---- phase 2: recycle rounds -- free a batch, immediately re-request
    # the SAME multiset of sizes (so several same-size free blocks coexist),
    # work the newcomers hard, and lightly re-touch a few still-alive ships
    # for cache realism. Weave in trap bursts. -------------------------------
    K = max(3, R // 6)
    burst_every = max(1, rounds // max(1, n_bursts)) if n_bursts else 0
    bursts_done = 0
    for rnd in range(rounds):
        ids_now = list(alive.keys())
        rng.shuffle(ids_now)
        batch = ids_now[:min(K, len(ids_now))]

        # group the about-to-be-freed batch by size; within each size class
        # of >=2 members, warm roughly half (heavy touches right before
        # freeing) and leave the rest cold -- chosen AFTER random batch
        # selection, so warmth is deliberately decoupled from each block's
        # (build-time) address.
        by_size = {}
        for i in batch:
            by_size.setdefault(alive[i], []).append(i)
        for sz, members in by_size.items():
            if len(members) < 2:
                continue
            # members sorted ascending by id ~ ascending build-time address
            # under a consistent lowest-address policy; deliberately keep the
            # LOWEST-id member cold (the one a lowest-address reflex will
            # reach for) and warm one of the higher-id members instead, so
            # the address-order heuristic is systematically wrong here.
            members_sorted = sorted(members)
            rest = members_sorted[1:]
            rng.shuffle(rest)
            warm_n = max(1, len(rest) // 2)
            for wid in rest[:warm_n]:
                for _ in range(rng.randint(5, 9)):
                    events.append(('T', wid, rng.randrange(sz)))

        freed_sizes = [alive[i] for i in batch]
        for i in batch:
            events.append(('F', i)); del alive[i]
        rng.shuffle(freed_sizes)
        for sz in freed_sizes:
            events.append(('A', next_id, sz))
            alive[next_id] = sz
            for _ in range(rng.randint(3, 6)):
                events.append(('T', next_id, rng.randrange(sz)))
            next_id += 1
        # mild ambient touches on the rest of the fleet
        rest = list(alive.keys())
        rng.shuffle(rest)
        for i in rest[:min(len(rest), K)]:
            events.append(('T', i, rng.randrange(alive[i])))

        if n_bursts and bursts_done < n_bursts and (rnd + 1) % burst_every == 0:
            next_id = insert_burst(rng, events, alive, next_id, bursts_done)
            bursts_done += 1

    while bursts_done < n_bursts:
        next_id = insert_burst(rng, events, alive, next_id, bursts_done)
        bursts_done += 1

    # ---- phase 3: drain everything still moored ----------------------------
    for i in list(alive.keys()):
        events.append(('F', i)); del alive[i]

    out = [f"{HEAP} {PAGE} {TLB} {len(events)}"]
    for ev in events:
        if ev[0] == 'A':
            out.append(f"A {ev[1]} {ev[2]}")
        elif ev[0] == 'F':
            out.append(f"F {ev[1]}")
        else:
            out.append(f"T {ev[1]} {ev[2]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

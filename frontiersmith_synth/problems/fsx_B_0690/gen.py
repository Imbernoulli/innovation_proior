#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE town-meeting bundling instance to stdout.

Town meeting: P public-works projects, each with a welfare weight, are put
before V voters.  Every voter has an explicit signed valuation for each
project: +s_i if she SUPPORTS project i, -o_i if she OPPOSES it, 0 if she is
neutral.  The town clerk must partition all P projects into at most K
referendum bundles.  A bundle passes iff a STRICT MAJORITY of voters have a
positive SUM of valuations across every project in that bundle (their
valuations for the bundled projects, added together -- a voter who is
neutral on every project in the bundle, or whose supports and opposes
cancel out to <=0, does not count as a yes).  The score is the total welfare
weight of projects sitting in a bundle that passes.

Planted structure (the trap):
  * A handful of "poison" projects are opposed by every single voter with an
    overwhelming magnitude -- no bundle that contains one can ever pass, and
    dropping one into an otherwise-fine bundle sinks the whole bundle.
  * One safe "consensus" project always passes alone by a wide margin.
  * Several broadly-popular "amenity" projects each individually clear a
    majority on their own (any way you combine them with other non-poison
    projects, they stay safe -- they never need rescuing).
  * The interesting cases are POLARIZING projects: each one, alone, falls
    just short of a majority (it has real supporters but not enough of
    them). For every polarizing project there exists somewhere among the
    OTHER small-weight projects a "sweetener" whose supporter set is drawn
    from voters who are NEUTRAL under the polarizer -- bundling that
    specific sweetener with the polarizer adds exactly enough fresh yes
    votes to cross the threshold. The polarizer's opposition magnitude is
    calibrated to be larger than any amenity/consensus/other-polarizer
    support magnitude, so bundling a polarizer with an unrelated popular
    project (the "cluster by average approval" instinct) never rescues it --
    only a bundle-mate whose supporter set actually reaches the polarizer's
    neutral pool with enough fresh votes does.

STDOUT format:
    line 1: P V K
    then, for each project i = 0..P-1, two lines:
      line A: w_i s_i o_i ns_i no_i
      line B: ns_i supporter voter indices (may be an empty line)
      line C: no_i opposer voter indices (may be an empty line)
"""
import sys, random

N_PAIRS  = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
N_POISON = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
P_TOTAL  = [8, 10, 13, 15, 18, 20, 23, 25, 28, 32]
V_TOTAL  = [41, 61, 91, 111, 141, 161, 191, 221, 261, 321]

OP_MAG = 20          # polarizer's opposition magnitude to its non-supporter free-pool voters
MAX_OTHER_S = 7       # cap on support magnitude for consensus/amenity/pair projects (< OP_MAG)


def build(t):
    rng = random.Random(96900 + 71 * t)
    idx = (t - 1) % 10
    n_pairs = N_PAIRS[idx]
    n_poison = N_POISON[idx]
    P = P_TOTAL[idx]
    V = V_TOTAL[idx]
    K = min(8, n_pairs + 3)

    reserved_size = max(8, V // 15)
    n_amenity = P - 1 - n_poison - 2 * n_pairs
    assert n_amenity >= 0, (t, P, n_poison, n_pairs)

    all_voters = list(range(V))
    # reserved blocks: one disjoint slice per pair, taken from the front of the range
    reserved_blocks = []
    cursor = 0
    for _ in range(n_pairs):
        block = list(range(cursor, cursor + reserved_size))
        reserved_blocks.append(block)
        cursor += reserved_size
    reserved_union = set()
    for b in reserved_blocks:
        reserved_union |= set(b)
    free_pool = [v for v in all_voters if v not in reserved_union]
    assert len(free_pool) >= V // 2 + 5, (t, len(free_pool), V)

    poison_projects = []
    for _ in range(n_poison):
        w = rng.randint(25, 40)
        poison_projects.append((w, 0, 1_000_000, [], list(all_voters)))

    # polarizer/sweetener pairs
    pair_projects = []
    for pi in range(n_pairs):
        block = reserved_blocks[pi]
        d = rng.randint(3, max(3, reserved_size - 3))
        p_support_n = max(1, V // 2 - d)
        p_support = rng.sample(free_pool, min(len(free_pool), p_support_n))
        p_support_set = set(p_support)
        p_oppose = [v for v in free_pool if v not in p_support_set]
        s_p = rng.randint(3, MAX_OTHER_S)
        w_p = rng.randint(18, 32)
        pair_projects.append((w_p, s_p, OP_MAG, sorted(p_support), sorted(p_oppose)))

        s_q = rng.randint(3, MAX_OTHER_S)
        w_q = rng.randint(2, 6)
        pair_projects.append((w_q, s_q, 0, sorted(block), []))

    # amenity/decoy projects: broadly popular, always pass alone, never need rescue
    amenity_projects = []
    for _ in range(n_amenity):
        margin = rng.randint(3, 10)
        supp_n = min(len(free_pool), V // 2 + margin)
        supp = rng.sample(free_pool, supp_n)
        s = rng.randint(3, MAX_OTHER_S)
        w = rng.randint(5, 14)
        amenity_projects.append((w, s, 0, sorted(supp), []))

    # project 0: safe consensus, comfortable majority drawn from the free pool.
    # Its weight is scaled to the total welfare reachable via pairs+amenities
    # (R) so the checker's baseline (= this weight alone, since the poisoned
    # lump always fails) stays a meaningful fraction of what a strong
    # partition can capture -- this keeps scoring headroom open instead of
    # saturating the moment a few pairs get rescued.
    R = sum(w for (w, s, o, sup, opp) in pair_projects + amenity_projects)
    w0 = max(10, round(R / 5.0))
    cons_support = rng.sample(free_pool, min(len(free_pool), V // 2 + 15))
    consensus = (w0, 5, 0, sorted(cons_support), [])

    # NOTE: project index 0 is always the safe consensus project (used by the
    # checker's internal baseline construction) -- order is otherwise
    # immaterial to scoring (bundle membership is a SET), so it is left as
    # generated (consensus, poison, pairs, amenities) rather than shuffled.
    projects = [consensus] + poison_projects + pair_projects + amenity_projects
    assert len(projects) == P
    return P, V, K, projects


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    P, V, K, projects = build(t)
    out = [f"{P} {V} {K}"]
    for (w, s, o, sup, opp) in projects:
        out.append(f"{w} {s} {o} {len(sup)} {len(opp)}")
        out.append(" ".join(map(str, sup)))
        out.append(" ".join(map(str, opp)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

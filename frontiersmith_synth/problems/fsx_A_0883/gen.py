import random
import sys

# Instance family: "hub + member" alphabet with a hidden, scrambled predecessor structure.
#
# The alphabet {0,...,k-1} is secretly partitioned (via a random relabeling, so the
# partition is NOT visible from symbol id order) into D "hub" symbols and the rest
# "member" symbols split into D classes, class d having roughly mc/D members. The
# string is a sequence of (hub, member) pairs: pick hub h_d, emit it, then emit ONE
# member drawn uniformly from class d's member pool, then pick the next hub.  Every
# occurrence of every class-d member is therefore ALWAYS immediately preceded by h_d --
# a strong, exploitable regularity that is invisible under the identity alphabet order
# (member ids are scrambled across the whole alphabet) but recoverable by clustering
# symbols by which symbol most often precedes them.

# Per-testId scale/seed table: (n, D, member_count, rng_seed).  Sizes increase mildly
# (small -> larger) while rng_seed is chosen (by prior offline search over many seeds)
# to give clean, sizeable exploitable structure at each scale -- these are the
# difficulty-ladder / trap cases: on every one of them, identity order (and simple
# statistics that are blind to WHICH specific symbol precedes another, e.g. sorting by
# raw occurrence frequency) leave most of the hub/member clustering unexploited; only
# clustering symbols by their actual dominant predecessor -- and then refining that
# clustering and the rotation -- recovers most of the achievable run reduction.
_TABLE = {
    1: (18, 2, 15, 22),
    2: (20, 2, 15, 17),
    3: (22, 2, 20, 23),
    4: (25, 2, 23, 14),
    5: (25, 2, 21, 23),
    6: (28, 2, 21, 7),
    7: (28, 2, 23, 14),
    8: (33, 2, 31, 21),
    9: (36, 2, 27, 8),
    10: (45, 2, 33, 1),
}


def gen_hub(seed, n_target, D, member_count, eps=0.0):
    rng = random.Random(seed)
    k = D + member_count
    roles = list(range(k))
    rng.shuffle(roles)
    hubs = roles[:D]
    members = roles[D:]
    rng.shuffle(members)
    classes = [[] for _ in range(D)]
    for i, s in enumerate(members):
        classes[i % D].append(s)
    cur_hub_idx = rng.randrange(D)
    seq = []
    while len(seq) < n_target:
        seq.append(hubs[cur_hub_idx])
        if rng.random() < eps:
            s = rng.choice(members)
        else:
            s = rng.choice(classes[cur_hub_idx])
        seq.append(s)
        cur_hub_idx = rng.randrange(D)
    seq = seq[:n_target]
    return seq, k


def main():
    test_id = int(sys.argv[1])
    n, D, mc, seed = _TABLE[test_id]
    seq, k = gen_hub(seed, n, D, mc, eps=0.0)
    print(len(seq), k)
    print(" ".join(map(str, seq)))


if __name__ == "__main__":
    main()

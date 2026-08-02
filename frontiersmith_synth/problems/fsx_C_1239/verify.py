#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for checksum-placement-design.

The participant's artifact is a routing of every message-bit position to one
of K parity ("checksum") groups. A held-out set of NPAT corruption events
(scattered independent flips, and bursts drawn from the channel's published
length distribution) is regenerated HERE, deterministically, from the SEED
value that is stored in the instance itself plus a private salt that never
appears in gen.py or in any file the solver can read. An event is caught iff
at least one group's assigned bits see an odd number of flips (XOR parity
mismatch). Score = participant's catch rate over the internal baseline's
catch rate (a single global parity bit covering the whole message).
"""
import random
import re
import sys

NPAT = 500          # size of the hidden evaluation set (fixed, not printed anywhere)
_SALT = 0x5BD1E995   # private constant -- never written to gen.py's output
INT_RE = re.compile(r"^-?\d+$")


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_tokens(path):
    try:
        with open(path) as f:
            return f.read().split()
    except OSError as e:
        fail("cannot read %s: %s" % (path, e))


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    itoks = read_tokens(in_path)
    if len(itoks) < 4:
        fail("truncated instance")
    idx = 0

    def inext():
        nonlocal idx
        if idx >= len(itoks):
            fail("truncated instance")
        v = itoks[idx]
        idx += 1
        return v

    M = int(inext())
    K = int(inext())
    seed = int(inext())
    pburst = float(inext())
    if M <= 0 or K <= 0 or not (0.0 <= pburst <= 1.0):
        fail("malformed instance header")

    nbl = int(inext())
    if nbl <= 0:
        fail("malformed length table")
    lengths, weights = [], []
    for _ in range(nbl):
        L = int(inext())
        W = int(inext())
        if L <= 0 or L >= M or W <= 0:
            fail("malformed length-table entry")
        lengths.append(L)
        weights.append(W)

    # ---------- parse + validate the participant's artifact strictly ----------
    otoks = read_tokens(out_path)
    if len(otoks) != M:
        fail("expected exactly %d group ids, got %d" % (M, len(otoks)))

    assign = [0] * M
    for i, tok in enumerate(otoks):
        if not INT_RE.match(tok):
            fail("token %r at position %d is not a plain integer (nan/inf/float rejected)" % (tok, i))
        v = int(tok)
        if not (0 <= v < K):
            fail("group id %d at position %d out of range [0,%d)" % (v, i, K))
        assign[i] = v

    # ---------- regenerate the hidden held-out corruption events ----------
    hidden_seed = ((seed * 1000003 + 20260726) ^ _SALT) & 0xFFFFFFFF
    rng = random.Random(hidden_seed)
    events = []
    for _ in range(NPAT):
        if rng.random() < pburst:
            L = rng.choices(lengths, weights=weights, k=1)[0]
            s = rng.randrange(M)
            flips = [(s + i) % M for i in range(L)]
        else:
            r = rng.choice((1, 2, 3))
            flips = rng.sample(range(M), r)
        events.append(flips)

    def catch_rate(routing):
        caught = 0
        for flips in events:
            parity = [0] * K
            for p in flips:
                g = routing[p]
                parity[g] ^= 1
            if any(parity):
                caught += 1
        return caught / NPAT

    # internal baseline B: a single global parity bit covering the whole
    # message (every position routed to group 0; groups 1..K-1 unused).
    baseline_routing = [0] * M
    B = catch_rate(baseline_routing)
    F = catch_rate(assign)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()

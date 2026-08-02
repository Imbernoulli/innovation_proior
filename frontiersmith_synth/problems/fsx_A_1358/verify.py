import sys, re

INT_RE = re.compile(r"^[+-]?\d+$")


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_int(tok):
    if not INT_RE.match(tok):
        return None
    return int(tok)


def compute_borda(ballots, m):
    scores = [0] * m
    for ballot in ballots:
        for pos, cand in enumerate(ballot):
            scores[cand] += (m - 1 - pos)
    return scores


def score_rule_winner(ballots, w, m):
    totals = [0] * m
    for ballot in ballots:
        for pos, cand in enumerate(ballot):
            totals[cand] += w[pos]
    best = 0
    for c in range(1, m):
        if totals[c] > totals[best]:
            best = c
    return best


def runoff2_winner(ballots, m):
    scores = compute_borda(ballots, m)
    order = sorted(range(m), key=lambda c: (-scores[c], c))
    f1, f2 = order[0], order[1]
    v1 = v2 = 0
    for ballot in ballots:
        p1 = ballot.index(f1)
        p2 = ballot.index(f2)
        if p1 < p2:
            v1 += 1
        else:
            v2 += 1
    if v1 > v2:
        return f1
    elif v2 > v1:
        return f2
    return min(f1, f2)


def simulate(ballots, m, rule_type, weights):
    if rule_type == 0:
        return score_rule_winner(ballots, weights, m)
    return runoff2_winner(ballots, m)


def main():
    in_tokens = open(sys.argv[1]).read().split()
    out_tokens = open(sys.argv[2]).read().split()

    try:
        it = iter(in_tokens)
        m = int(next(it))
        n = int(next(it))
        rule_type = int(next(it))
        target = int(next(it))
        weights = None
        if rule_type == 0:
            weights = [int(next(it)) for _ in range(m)]
        ballots = []
        for _ in range(n):
            ballots.append([int(next(it)) for _ in range(m)])
    except Exception:
        fail("bad input file (harness bug)")

    # ---- internal baseline B: coalition = ALL n voters, every one rewritten to
    #      [target] + (remaining candidates in ascending index order). This always
    #      makes `target` the unique winner under both rule types (see statement). ----
    B = 1.0 / n

    # ---- parse participant output strictly ----
    if len(out_tokens) < 1:
        fail("empty output")
    k = parse_int(out_tokens[0])
    if k is None:
        fail("k not an integer")
    if k < 0 or k > n:
        fail("k out of range [0,%d]: %d" % (n, k))
    if k == 0:
        fail("empty coalition cannot change the winner (target != sincere winner by construction)")

    needed = k * (m + 1)
    rest = out_tokens[1:]
    if len(rest) != needed:
        fail("expected %d tokens for %d coalition lines, got %d" % (needed, k, len(rest)))

    coalition = {}
    for j in range(k):
        chunk = rest[j * (m + 1): (j + 1) * (m + 1)]
        vidx = parse_int(chunk[0])
        if vidx is None:
            fail("voter index not an integer")
        if vidx < 0 or vidx >= n:
            fail("voter index out of range: %d" % vidx)
        if vidx in coalition:
            fail("voter %d listed twice in coalition" % vidx)
        ballot_toks = chunk[1:]
        ballot = []
        for tok in ballot_toks:
            v = parse_int(tok)
            if v is None:
                fail("ballot entry not an integer: %r" % tok)
            ballot.append(v)
        if sorted(ballot) != list(range(m)):
            fail("voter %d's new ballot is not a permutation of 0..%d: %r" % (vidx, m - 1, ballot))
        coalition[vidx] = ballot

    # ---- build modified profile ----
    modified = list(ballots)
    for vidx, ballot in coalition.items():
        modified[vidx] = ballot

    winner = simulate(modified, m, rule_type, weights)
    if winner != target:
        fail("manipulation failed: winner=%d target=%d" % (winner, target))

    F = 1.0 / k
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("k=%d n=%d winner=%d target=%d Ratio: %.6f" % (k, n, winner, target, sc / 1000.0))


if __name__ == "__main__":
    main()

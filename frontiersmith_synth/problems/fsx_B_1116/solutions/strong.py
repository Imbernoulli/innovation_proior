# TIER: strong
# The insight: treat the CORPUS as the unit of design, not just this one
# cantus firmus, and treat the two global (whole-piece) hard rules -- unique
# climax, and the contrary-motion quota -- as constraints to PRE-COMMIT to
# before filling in the rest, not as things to hope a local pass stumbles
# into.
#
#  1. Corpus self-assignment: this program is invoked once per instance in a
#     totally isolated subprocess that never sees any other instance's
#     answer. But every instance publishes its own instance_id/corpus_size,
#     so the policy can still deliberately manage its OWN distribution of
#     climax positions across the whole corpus by picking
#     target_bucket = instance_id % climax_buckets -- a seeded, corpus-aware
#     role assignment with no communication needed. This is what earns the
#     corpus-level style-diversity bonus that a same-recipe-every-time
#     policy always collapses.
#  2. Climax pre-commitment: rather than discover a duplicate maximum after
#     the fact, fix a single (position, pitch) as the intended unique climax
#     FIRST (preferring a position inside the target bucket, and a pitch near
#     the top of the legal range), forbid every other position from reaching
#     that pitch, and only THEN search.
#  3. Global contrary-motion accounting via backtracking DFS: the search
#     state carries the running contrary-move count, and a step is pruned as
#     soon as it is mathematically impossible to reach the quota with the
#     moves remaining -- real planning against a whole-piece threshold,
#     not a one-step-lookback rule.
import sys, json


def solve(inst):
    cantus = inst["cantus"]
    L = len(cantus)
    lo, hi = inst["cp_range"]
    rules = inst["rules"]
    cc = set(rules["consonant_classes"])
    pc = set(rules["perfect_classes"])
    bc = set(rules["boundary_classes"])
    max_leap = rules["max_leap"]
    min_cf = rules["min_contrary_frac"]
    buckets = rules["climax_buckets"]
    iid = inst["instance_id"]

    target_bucket = iid % buckets

    def legal_positions(i, prev, prev_class):
        out = []
        for p in range(lo, hi + 1):
            vc = (p - cantus[i]) % 7
            if vc not in cc:
                continue
            if (i == 0 or i == L - 1) and vc not in bc:
                continue
            if prev is not None and abs(p - prev) > max_leap:
                continue
            if prev_class is not None and vc in pc and prev_class in pc and vc == prev_class:
                continue
            out.append((p, vc))
        return out

    def cat(d):
        d = abs(d)
        if d <= 1:
            return 0
        elif d <= 3:
            return 1
        else:
            return 2

    def try_build(climax_pos, climax_val):
        def domain(i, prev, prev_class):
            if i == climax_pos:
                vc = (climax_val - cantus[i]) % 7
                if vc not in cc:
                    return []
                if (i == 0 or i == L - 1) and vc not in bc:
                    return []
                if prev is not None and abs(climax_val - prev) > max_leap:
                    return []
                if prev_class is not None and vc in pc and prev_class in pc and vc == prev_class:
                    return []
                return [(climax_val, vc)]
            cands = legal_positions(i, prev, prev_class)
            return [(p, vc) for (p, vc) in cands if p < climax_val]

        def contrary_reachable(moves_contrary, moves_total, remaining):
            if moves_total + remaining == 0:
                return True
            best_possible = (moves_contrary + remaining) / (moves_total + remaining)
            return best_possible >= min_cf - 1e-9

        def dfs(i, prev, prev_class, moves_contrary, moves_total, path, used_cats):
            if i == L:
                frac = moves_contrary / moves_total if moves_total > 0 else 1.0
                if frac < min_cf - 1e-9:
                    return None
                return list(path)
            cands = domain(i, prev, prev_class)
            scored = []
            for (p, vc) in cands:
                if i > 0:
                    dcp = p - prev
                    dcf = cantus[i] - cantus[i - 1]
                    is_c = dcp != 0 and dcf != 0 and (dcp > 0) != (dcf > 0)
                else:
                    is_c = False
                c = cat(p - prev) if prev is not None else None
                div_gain = 0 if (c is None or c in used_cats) else 1
                pref = (1 if is_c else 0, div_gain, -abs(p - cantus[i]))
                scored.append((pref, p, vc, is_c, c))
            scored.sort(key=lambda t: t[0], reverse=True)
            for pref, p, vc, is_c, c in scored:
                nmc = moves_contrary + (1 if is_c else 0)
                nmt = moves_total + (1 if i > 0 else 0)
                remaining = (L - 1) - nmt
                if not contrary_reachable(nmc, nmt, remaining):
                    continue
                nu = used_cats | ({c} if c is not None else set())
                res = dfs(i + 1, p, vc, nmc, nmt, path + [p], nu)
                if res is not None:
                    return res
            return None

        return dfs(0, None, None, 0, 0, [], set())

    def bucket_of(idx):
        return min(buckets - 1, int(idx / (L - 1) * buckets)) if L > 1 else 0

    positions_by_bucket = {}
    for idx in range(L):
        positions_by_bucket.setdefault(bucket_of(idx), []).append(idx)

    pos_order = list(positions_by_bucket.get(target_bucket, []))
    for b in range(buckets):
        if b == target_bucket:
            continue
        pos_order += positions_by_bucket.get(b, [])

    val_candidates = list(range(hi, lo - 1, -1))[:14]

    result = None
    for pos in pos_order:
        for val in val_candidates:
            vc = (val - cantus[pos]) % 7
            if vc not in cc:
                continue
            if (pos == 0 or pos == L - 1) and vc not in bc:
                continue
            r = try_build(pos, val)
            if r is not None:
                result = r
                break
        if result is not None:
            break

    if result is not None:
        return {"cp": result}

    # Fallback (should essentially never trigger given generous consonance
    # density): plain nearest-tone construction, so we never emit a crash.
    cp = []
    prev = None
    prev_class = None
    for i in range(L):
        cands = legal_positions(i, prev, prev_class)
        if not cands:
            cands = [(cantus[i], 0)]
        best = min(cands, key=lambda pc_: (abs(pc_[0] - cantus[i]), pc_[0]))[0]
        cp.append(best)
        prev = best
        prev_class = (best - cantus[i]) % 7
    return {"cp": cp}


def main():
    inst = json.load(sys.stdin)
    print(json.dumps(solve(inst)))


main()

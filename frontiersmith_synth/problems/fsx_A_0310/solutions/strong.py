# TIER: strong
# Warm-started local search for the fuel-budgeted survey.  Start from the
# value-density insertion greedy, then alternate three move types until no pass
# improves:
#   (1) 2-OPT on the visited route to SHORTEN it (open path) -- freed fuel becomes
#       budget for more systems;
#   (2) cheapest-insertion of any unvisited system whose insertion still fits fuel,
#       preferring high prize-per-extra-fuel;
#   (3) SWAP: drop a low-value visited system and insert the best reachable unvisited
#       system if total collected value strictly rises.
# This reclaims value that one-shot greedy strands by re-ordering and revising -- but
# the visit-everything ceiling stays unaffordable, so scores keep headroom under 1.0.
import sys, json, math

inst = json.load(sys.stdin)
N, L = inst["N"], inst["L"]
x, y, p = inst["x"], inst["y"], inst["p"]
bx, by = inst["bx"], inst["by"]


def d(i, j):
    return math.hypot(x[i] - x[j], y[i] - y[j])


def db(j):
    return math.hypot(bx - x[j], by - y[j])


def path_len(path):
    if not path:
        return 0.0
    tot = db(path[0])
    for k in range(len(path) - 1):
        tot += d(path[k], path[k + 1])
    return tot


def path_val(path):
    return sum(p[j] for j in path)


def insertion(path, j):
    """Cheapest extra fuel + position to insert j into open path."""
    if not path:
        return db(j), 0
    best_extra = db(j) + d(j, path[0]) - db(path[0])
    best_pos = 0
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        extra = d(a, j) + d(j, b) - d(a, b)
        if extra < best_extra:
            best_extra = extra
            best_pos = k + 1
    extra_end = d(path[-1], j)
    if extra_end < best_extra:
        best_extra = extra_end
        best_pos = len(path)
    return best_extra, best_pos


def greedy_build():
    path = []
    length = 0.0
    used = set()
    while True:
        bj = -1
        bscore = 0.0
        bextra = 0.0
        bpos = 0
        for j in range(N):
            if j in used:
                continue
            extra, pos = insertion(path, j)
            if length + extra > L + 1e-9:
                continue
            denom = extra if extra > 1e-9 else 1e-9
            score = p[j] / denom
            if score > bscore + 1e-15:
                bscore = score
                bj = j
                bextra = extra
                bpos = pos
        if bj < 0:
            break
        path.insert(bpos, bj)
        length += bextra
        used.add(bj)
    return path


def two_opt(path):
    """Shorten the open path with 2-opt segment reversals."""
    n = len(path)
    if n < 3:
        return path
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            # node before segment start: starbase if i==0 else path[i-1]
            for k in range(i + 1, n):
                a_prev = path[i - 1] if i > 0 else None
                b = path[k]
                c = path[k + 1] if k + 1 < n else None
                # edges removed: (a_prev -> path[i]) and (path[k] -> c)
                # edges added:   (a_prev -> path[k]) and (path[i] -> c)
                pi = path[i]
                if a_prev is None:
                    old1 = db(pi)
                    new1 = db(b)
                else:
                    old1 = d(a_prev, pi)
                    new1 = d(a_prev, b)
                if c is None:
                    old2 = 0.0
                    new2 = 0.0
                else:
                    old2 = d(b, c)
                    new2 = d(pi, c)
                if new1 + new2 + 1e-9 < old1 + old2:
                    path[i:k + 1] = path[i:k + 1][::-1]
                    improved = True
    return path


def local_search():
    path = greedy_build()
    for _ in range(12):
        improved = False
        # (1) shorten
        before = path_len(path)
        path = two_opt(path)
        length = path_len(path)
        if length + 1e-6 < before:
            improved = True
        # (2) insert affordable systems (best density first)
        used = set(path)
        while True:
            bj = -1
            bscore = 0.0
            bextra = 0.0
            bpos = 0
            for j in range(N):
                if j in used:
                    continue
                extra, pos = insertion(path, j)
                if length + extra > L + 1e-9:
                    continue
                denom = extra if extra > 1e-9 else 1e-9
                score = p[j] / denom
                if score > bscore + 1e-15:
                    bscore = score
                    bj = j
                    bextra = extra
                    bpos = pos
            if bj < 0:
                break
            path.insert(bpos, bj)
            length += bextra
            used.add(bj)
            improved = True
        # (3) swap: drop a low-value visited system, insert a richer reachable one
        used = set(path)
        for idx in range(len(path)):
            drop = path[idx]
            trial = path[:idx] + path[idx + 1:]
            base_len = path_len(trial)
            bj = -1
            bgain = 0.0
            bextra = 0.0
            bpos = 0
            for j in range(N):
                if j in used or j == drop:
                    continue
                extra, pos = insertion(trial, j)
                if base_len + extra > L + 1e-9:
                    continue
                gain = p[j] - p[drop]
                if gain > bgain + 1e-9:
                    bgain = gain
                    bj = j
                    bextra = extra
                    bpos = pos
            if bj >= 0:
                trial.insert(bpos, bj)
                path = trial
                length = path_len(path)
                used = set(path)
                improved = True
        if not improved:
            break
    return path


best = local_search()
print(json.dumps({"route": best}))

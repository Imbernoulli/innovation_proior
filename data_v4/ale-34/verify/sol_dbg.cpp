#include <bits/stdc++.h>
using namespace std;

/*
  ale-34  Maze Carving to a Target Difficulty.

  We are given an H x W grid of walls '#' and open cells '.', a start S and an
  end T (both open), and a budget B. We must carve exactly B currently-wall
  cells into open. The score is the BFS shortest-path length S->T over the
  resulting open cells (0 if S and T end up disconnected, or if the carve set is
  invalid). We MAXIMIZE that shortest path.

  Idea.
    * A feasible solution must (a) carve exactly B distinct wall cells and
      (b) keep S and T connected. We first secure feasibility with a minimum
      carve connecting path, then pad to exactly B.
    * To make the shortest path LONG, we want the carved open region to behave
      like one long, winding corridor with NO shortcuts. The cleanest way to get
      "no shortcut" is to make the S->T corridor an INDUCED path: a sequence of
      open cells where consecutive cells are adjacent and non-consecutive cells
      are never adjacent. For an induced path the geodesic equals the full path
      length, so maximizing path length maximizes the score.
    * Construction: a randomized greedy self-avoiding walk from S that grows an
      induced path, biased to wander (away from T) while budget remains, then
      homes in on T. We keep the best feasible carve set over many restarts.
    * Refinement (the innovation): simulated annealing whose move swaps one
      carved cell for an adjacent uncarved wall. We re-run BFS to re-score a
      candidate ONLY when the swap touches the current geodesic corridor;
      a swap disjoint from the geodesic that does not create a new short edge
      cannot shorten the geodesic, so it is accepted/rejected by a cheap local
      test instead of a full BFS. This keeps thousands of moves/sec feasible.

  We never emit an infeasible solution: a guaranteed connecting carve set is
  computed up front and used as a fallback, and we always print exactly B
  distinct wall cells.
*/

static int H, W, B;
static int SR, SC, TR, TC;
static vector<string> G0;     // original grid
static int N;                 // H*W

static inline int id(int r, int c) { return r * W + c; }
static const int DR[4] = {1, -1, 0, 0};
static const int DC[4] = {0, 0, 1, -1};

static std::mt19937 rng(998244353u);
static inline int randint(int a, int b) { // inclusive
    return a + (int)(rng() % (unsigned)(b - a + 1));
}

// BFS shortest path on a boolean "open" grid. Returns -1 if unreachable.
static int bfsDist(const vector<char>& open) {
    static vector<int> dist;
    dist.assign(N, -1);
    deque<int> dq;
    dist[id(SR, SC)] = 0;
    dq.push_back(id(SR, SC));
    int target = id(TR, TC);
    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        if (u == target) return dist[u];
        int r = u / W, c = u % W;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            if (open[v] && dist[v] < 0) {
                dist[v] = dist[u] + 1;
                dq.push_back(v);
            }
        }
    }
    return -1;
}

// Minimum-carve connecting path: 0/1 BFS where stepping onto a wall costs 1
// (we must carve it) and onto an open cell costs 0. Returns the set of wall
// cells that must be carved to connect S and T with the fewest carves, or empty
// vector with feasible=false if T is unreachable even by carving (cannot happen
// on a full grid, but we guard anyway).
static vector<int> minCarveConnect(const vector<char>& isWall, bool& feasible) {
    vector<int> dist(N, INT_MAX), par(N, -1);
    deque<int> dq;
    int s = id(SR, SC), t = id(TR, TC);
    dist[s] = 0;
    dq.push_back(s);
    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        int r = u / W, c = u % W;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            int w = isWall[v] ? 1 : 0;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                par[v] = u;
                if (w == 0) dq.push_front(v);
                else dq.push_back(v);
            }
        }
    }
    vector<int> carve;
    if (dist[t] == INT_MAX) { feasible = false; return carve; }
    feasible = true;
    int u = t;
    while (u != -1) {
        if (isWall[u]) carve.push_back(u);
        u = par[u];
    }
    return carve; // the walls we must open along the cheapest path
}

// Greedy induced-path construction from S. Grows a self-avoiding walk; a cell is
// addable only if, ignoring the current path tail it would attach to, it is not
// adjacent to any already-chosen path cell except its predecessor (keeps the
// path induced => no shortcuts). Wanders while budget allows, then heads to T.
// Returns the chosen open-cell set "onPath" (path cells incl. pre-open used),
// and the carved-wall list; or signals failure.
struct BuildResult {
    bool ok = false;
    vector<int> carve;       // wall cells carved (size may be < or up to budget)
    int pathLen = -1;        // induced path length S..T
};

static BuildResult buildInducedPath(const vector<char>& isWall, int budget) {
    BuildResult R;
    vector<char> onPath(N, 0);
    vector<int> path;
    int s = id(SR, SC), t = id(TR, TC);
    onPath[s] = 1;
    path.push_back(s);
    int carved = 0; // walls opened so far on path
    int cur = s;

    auto manhT = [&](int u) {
        int r = u / W, c = u % W;
        return abs(r - TR) + abs(c - TC);
    };

    // A neighbor `v` of `cur` can extend the induced path if:
    //  - v in bounds, not already on path, v != predecessor;
    //  - carving v (if wall) stays within budget;
    //  - v is not adjacent to any path cell other than cur (induced).
    auto canExtend = [&](int v, int& addCarve) -> bool {
        if (onPath[v]) return false;
        int r = v / W, c = v % W;
        addCarve = isWall[v] ? 1 : 0;
        if (carved + addCarve > budget) return false;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int u = id(nr, nc);
            if (u == cur) continue;
            if (onPath[u]) return false; // would create a chord => shortcut
        }
        return true;
    };

    int safety = 0;
    int maxSteps = 8 * N;
    while (cur != t && safety++ < maxSteps) {
        // Decide whether to wander (spend budget, increase length) or to home
        // toward T. Home more strongly as remaining budget shrinks.
        int remaining = budget - carved;
        bool homing = (remaining <= manhT(cur) + 1) || (randint(0, 99) < 12);

        // Gather extendable neighbors.
        int cr = cur / W, cc = cur % W;
        int bestV = -1, bestCarve = 0;
        // Order candidates; when homing prefer reducing distance to T, when
        // wandering prefer increasing it. Add randomness to diversify restarts.
        int order[4] = {0, 1, 2, 3};
        for (int i = 3; i > 0; i--) std::swap(order[i], order[randint(0, i)]);
        long long bestKey = LLONG_MIN;
        for (int oi = 0; oi < 4; oi++) {
            int k = order[oi];
            int nr = cr + DR[k], nc = cc + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            int addCarve;
            if (!canExtend(v, addCarve)) continue;
            int d = manhT(v);
            long long key = homing ? (long long)(-d) * 4 + randint(0, 3)
                                   : (long long)(d) * 4 + randint(0, 3);
            if (key > bestKey) { bestKey = key; bestV = v; bestCarve = addCarve; }
        }
        if (bestV < 0) break; // stuck: cannot extend induced path further
        onPath[bestV] = 1;
        path.push_back(bestV);
        carved += bestCarve;
        cur = bestV;
        if (cur == t) break;
    }

    if (cur != t) return R; // did not reach T as an induced path

    // Success: collect carved walls.
    R.ok = true;
    R.pathLen = (int)path.size() - 1;
    for (int u : path) if (isWall[u]) R.carve.push_back(u);
    return R;
}

// Pad a carve set up to exactly B distinct wall cells, choosing extra walls that
// (heuristically) do not create a shortcut: prefer walls far from the start-end
// geodesic and not adjacent to >=2 currently-open cells. As a guarantee of
// feasibility we will, if necessary, accept any remaining wall.
static void padToB(vector<int>& carve, const vector<char>& isWall) {
    vector<char> chosen(N, 0);
    for (int u : carve) chosen[u] = 1;

    // current open set = pre-open OR chosen-to-carve
    auto buildOpen = [&]() {
        vector<char> open(N, 0);
        for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
        return open;
    };

    // Candidate walls (not yet chosen).
    vector<int> cands;
    for (int u = 0; u < N; u++) if (isWall[u] && !chosen[u]) cands.push_back(u);

    // Rank candidates by fewest open-neighbors (less likely to add a shortcut),
    // then by distance from both S and T.
    auto openNeighbors = [&](const vector<char>& open, int u) {
        int r = u / W, c = u % W, cnt = 0;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            if (open[id(nr, nc)]) cnt++;
        }
        return cnt;
    };

    while ((int)carve.size() < B) {
        vector<char> open = buildOpen();
        int curDist = bfsDist(open);
        int pick = -1; long long bestKey = LLONG_MIN; int pickIdx = -1;
        for (int i = 0; i < (int)cands.size(); i++) {
            int u = cands[i];
            if (chosen[u]) continue;
            int onb = openNeighbors(open, u);
            // Prefer isolated walls (0 or 1 open neighbor): opening them cannot
            // bridge two corridor strands, so it cannot shorten the geodesic.
            long long key = -(long long)onb * 1000;
            int r = u / W, c = u % W;
            key -= abs(r - SR) + abs(c - SC) + abs(r - TR) + abs(c - TC);
            if (key > bestKey) { bestKey = key; pick = u; pickIdx = i; }
        }
        if (pick < 0) break; // no candidate left (shouldn't happen if B<=walls)
        // Only accept if it does not reduce the geodesic; if it does and we still
        // need to pad, accept anyway (feasibility first) but try others first.
        chosen[pick] = 1;
        carve.push_back(pick);
        // erase from cands fast
        cands[pickIdx] = cands.back(); cands.pop_back();
    }
}

// ---- Simulated annealing refinement (the innovation) -----------------------
// State: a set of carved walls (chosen[]). Move: swap one carved wall OUT for an
// adjacent uncarved wall IN (keeps |carve| == B). Score = bfsDist of resulting
// open grid (longer is better). We only do a full BFS re-score when the swap is
// "near" the current geodesic; otherwise we keep the score unchanged (a swap
// that neither removes a geodesic cell nor opens a wall adjacent to two open
// strands cannot change the geodesic length). On any swap that disconnects S/T
// we reject.
static void anneal(vector<int>& carve, const vector<char>& isWall,
                   double timeLimitSec) {
    auto t0 = chrono::steady_clock::now();
    vector<char> chosen(N, 0);
    for (int u : carve) chosen[u] = 1;

    auto buildOpen = [&]() {
        vector<char> open(N, 0);
        for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
        return open;
    };

    // Compute geodesic membership: which open cells lie on SOME shortest path.
    // dS = dist from S, dT = dist from T; cell on a geodesic iff
    // dS+dT == totalDist and both finite.
    auto bfsFrom = [&](int src, const vector<char>& open, vector<int>& dist) {
        dist.assign(N, -1);
        deque<int> dq; dist[src] = 0; dq.push_back(src);
        while (!dq.empty()) {
            int u = dq.front(); dq.pop_front();
            int r = u / W, c = u % W;
            for (int k = 0; k < 4; k++) {
                int nr = r + DR[k], nc = c + DC[k];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int v = id(nr, nc);
                if (open[v] && dist[v] < 0) { dist[v] = dist[u] + 1; dq.push_back(v); }
            }
        }
    };

    vector<char> open = buildOpen();
    vector<int> dS, dT;
    bfsFrom(id(SR, SC), open, dS);
    bfsFrom(id(TR, TC), open, dT);
    int curScore = dS[id(TR, TC)];
    if (curScore < 0) curScore = 0;

    vector<char> onGeo(N, 0);
    auto recomputeGeo = [&]() {
        open = buildOpen();
        bfsFrom(id(SR, SC), open, dS);
        bfsFrom(id(TR, TC), open, dT);
        int total = dS[id(TR, TC)];
        curScore = (total < 0) ? 0 : total;
        onGeo.assign(N, 0);
        if (total >= 0) {
            for (int u = 0; u < N; u++)
                if (open[u] && dS[u] >= 0 && dT[u] >= 0 && dS[u] + dT[u] == total)
                    onGeo[u] = 1;
        }
    };
    recomputeGeo();

    vector<int> bestCarve = carve;
    int bestScore = curScore;

    // Quick test: does opening wall `u` create a "new adjacency" between two open
    // cells that are far apart along the path (a potential shortcut)? We treat
    // any wall with >= 2 open neighbors as geodesic-relevant (could shortcut).
    auto openNbrCount = [&](const vector<char>& op, int u) {
        int r = u / W, c = u % W, cnt = 0;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            if (op[id(nr, nc)]) cnt++;
        }
        return cnt;
    };

    double T = 3.0, Tend = 0.05;
    long long iters = 0;
    while (true) {
        if ((iters & 1023) == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            if (el > timeLimitSec) break;
            double frac = el / timeLimitSec;
            T = 3.0 * pow(Tend / 3.0, frac);
        }
        iters++;

        if (carve.empty()) break;
        // Pick a carved wall to remove and an adjacent uncarved wall to add.
        int idxOut = randint(0, (int)carve.size() - 1);
        int wOut = carve[idxOut];
        int ro = wOut / W, co = wOut % W;
        // pick an in-cell: an adjacent wall not currently carved, or a random wall.
        int wIn = -1;
        {
            int ord[4] = {0,1,2,3};
            for (int i = 3; i > 0; i--) std::swap(ord[i], ord[randint(0,i)]);
            for (int oi = 0; oi < 4; oi++) {
                int k = ord[oi];
                int nr = ro + DR[k], nc = co + DC[k];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int v = id(nr, nc);
                if (isWall[v] && !chosen[v]) { wIn = v; break; }
            }
        }
        if (wIn < 0) {
            // fallback: random wall not chosen
            int tries = 0;
            while (tries++ < 8) {
                int v = randint(0, N - 1);
                if (isWall[v] && !chosen[v] && v != wOut) { wIn = v; break; }
            }
        }
        if (wIn < 0) continue;

        // Apply swap tentatively.
        chosen[wOut] = 0; chosen[wIn] = 1;

        // Decide whether the swap can affect the geodesic cheaply.
        // It is geodesic-relevant if wOut was ON the current geodesic, OR if
        // opening wIn could create a shortcut (>=2 open neighbors in the new
        // grid). Otherwise the geodesic length is unchanged.
        bool relevant = onGeo[wOut];
        if (!relevant) {
            // Build a light open view to count neighbors of wIn after swap.
            // (open[] currently reflects the pre-swap grid; adjust on the fly.)
            // Count open neighbors of wIn under the swapped grid.
            int r = wIn / W, c = wIn % W, cnt = 0;
            for (int k = 0; k < 4; k++) {
                int nr = r + DR[k], nc = c + DC[k];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int v = id(nr, nc);
                bool vopen = (!isWall[v]) || chosen[v];
                if (vopen) cnt++;
            }
            if (cnt >= 2) relevant = true;
        }

        if (!relevant) {
            // Cheap accept: geodesic unchanged (still connected, same length).
            // wOut not on geodesic and removing it cannot disconnect S-T because
            // it was not needed by the shortest path; we keep curScore/onGeo but
            // must refresh onGeo lazily — mark wOut closed in open view.
            // To stay exact and safe we periodically refresh; here we just keep
            // the score and update the open membership flags minimally.
            open[wOut] = (!isWall[wOut]); // wOut now closed unless pre-open
            open[wIn] = true;
            onGeo[wOut] = 0;
            // accept (no score change)
            continue;
        }

        // Full re-score.
        vector<char> nopen = buildOpen();
        int total = bfsDist(nopen);
        int newScore = (total < 0) ? -1 : total;

        bool accept;
        if (newScore < 0) {
            accept = false; // disconnected -> never accept
        } else if (newScore >= curScore) {
            accept = true;
        } else {
            double d = curScore - newScore;
            double prob = exp(-d / std::max(1e-9, T));
            accept = (randint(0, 1000000) / 1000000.0) < prob;
        }

        if (accept) {
            carve[idxOut] = wIn; // keep the carve vector in sync
            recomputeGeo();
            if (curScore > bestScore) { bestScore = curScore; bestCarve = carve; }
        } else {
            // revert
            chosen[wOut] = 1; chosen[wIn] = 0;
        }
    }

    carve = bestCarve;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> H >> W >> B)) return 0;
    cin >> SR >> SC >> TR >> TC;
    G0.assign(H, string());
    for (int r = 0; r < H; r++) {
        string line;
        cin >> line;
        // pad/truncate to width W defensively
        if ((int)line.size() < W) line.resize(W, '.');
        G0[r] = line;
    }
    N = H * W;

    vector<char> isWall(N, 0);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            isWall[id(r, c)] = (G0[r][c] == '#') ? 1 : 0;
    // S and T are open by construction.
    isWall[id(SR, SC)] = 0;
    isWall[id(TR, TC)] = 0;

    int totalWalls = 0;
    for (int u = 0; u < N; u++) totalWalls += isWall[u];
    if (B > totalWalls) B = totalWalls; // cannot carve more than available

    // ---- Guaranteed feasible fallback: minimum-carve connecting path -------
    bool feasible = false;
    vector<int> fallback = minCarveConnect(isWall, feasible);
    // fallback connects S-T using <= ... walls; pad it to exactly B.

    // ---- Strong construction: best induced path over restarts -------------
    vector<int> best;
    int bestLen = -1;
    int restarts = 0;
    auto t0 = chrono::steady_clock::now();
    double buildBudget = 0.45; // seconds for construction phase
    while (true) {
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (restarts >= 4 && el > buildBudget) break;
        if (el > 0.8) break;
        restarts++;
        BuildResult R = buildInducedPath(isWall, B);
        if (R.ok && R.pathLen > bestLen && (int)R.carve.size() <= B) {
            bestLen = R.pathLen;
            best = R.carve;
        }
        if (restarts > 400) break;
    }

    fprintf(stderr, "[dbg] restarts=%d bestLen=%d bestCarveSize=%zu B=%d feasible=%d fallbackSize=%zu\n", restarts, bestLen, best.size(), B, (int)feasible, fallback.size());
    vector<int> carve;
    if (!best.empty() || bestLen >= 0) {
        carve = best;
    } else if (feasible) {
        carve = fallback;
    } else {
        carve.clear();
    }

    // Pad up to exactly B distinct wall cells (feasibility-preserving padding).
    // First make sure we are connected; if the induced path is empty/short and
    // we have a feasible fallback, ensure the fallback walls are included.
    {
        vector<char> chosen(N, 0);
        for (int u : carve) chosen[u] = 1;
        // verify connectivity; if not connected, merge fallback in.
        {
            vector<char> open(N, 0);
            for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
            if (bfsDist(open) < 0 && feasible) {
                for (int u : fallback) if (!chosen[u]) { chosen[u] = 1; }
                carve.clear();
                for (int u = 0; u < N; u++) if (isWall[u] && chosen[u]) carve.push_back(u);
            }
        }
        // dedupe / clamp
        chosen.assign(N, 0);
        vector<int> dedup;
        for (int u : carve) if (isWall[u] && !chosen[u]) { chosen[u] = 1; dedup.push_back(u); }
        carve = dedup;
        if ((int)carve.size() > B) carve.resize(B); // shouldn't normally happen
    }

    padToB(carve, isWall);

    // Final safety: if still < B (e.g. degenerate), add any remaining walls.
    if ((int)carve.size() < B) {
        vector<char> chosen(N, 0);
        for (int u : carve) chosen[u] = 1;
        for (int u = 0; u < N && (int)carve.size() < B; u++)
            if (isWall[u] && !chosen[u]) { chosen[u] = 1; carve.push_back(u); }
    }
    if ((int)carve.size() > B) carve.resize(B);

    // If somehow disconnected after padding, fall back to the guaranteed
    // connecting set padded up (never emit an infeasible solution).
    {
        vector<char> chosen(N, 0);
        for (int u : carve) chosen[u] = 1;
        vector<char> open(N, 0);
        for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
        if (bfsDist(open) < 0) {
            // rebuild from fallback
            vector<char> ch2(N, 0);
            vector<int> c2;
            for (int u : fallback) if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            // pad
            for (int u = 0; u < N && (int)c2.size() < B; u++)
                if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            if ((int)c2.size() > B) c2.resize(B);
            carve = c2;
        }
    }

    // ---- Refinement: simulated annealing (the innovation) -----------------
    {
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        double remain = 1.85 - el; // total budget ~1.9s; leave slack for I/O
        if (remain > 0.05 && (int)carve.size() == B)
            anneal(carve, isWall, remain);
    }

    // Final feasibility guard after SA.
    {
        vector<char> chosen(N, 0);
        bool bad = ((int)carve.size() != B);
        for (int u : carve) {
            if (u < 0 || u >= N || !isWall[u] || chosen[u]) { bad = true; break; }
            chosen[u] = 1;
        }
        if (!bad) {
            vector<char> open(N, 0);
            for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
            if (bfsDist(open) < 0) bad = true;
        }
        if (bad) {
            // rebuild guaranteed feasible from fallback padded to B
            vector<char> ch2(N, 0);
            vector<int> c2;
            for (int u : fallback) if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            for (int u = 0; u < N && (int)c2.size() < B; u++)
                if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            if ((int)c2.size() > B) c2.resize(B);
            carve = c2;
        }
    }

    // Output: B lines "r c".
    string out;
    out.reserve(carve.size() * 8);
    for (int u : carve) {
        int r = u / W, c = u % W;
        out += to_string(r);
        out += ' ';
        out += to_string(c);
        out += '\n';
    }
    cout << out;
    return 0;
}

// Interval Scheduling on Few Rooms (ale-25)
//
// Assign each of n weighted intervals to one of K identical rooms, or reject
// it. Intervals in the same room may not overlap (half-open [s,e): touching
// endpoints is allowed). Maximize the total weight of accepted intervals.
//
// Read instance from stdin:   n K  then n lines: s e w
// Write solution to stdout:   n lines, the i-th is room id in [0,K) or -1.
//
// Method (the lever): treat each room as a resource and keep, per room, a
// balanced ordered set of the currently-placed intervals keyed by start time.
// This makes the only feasibility question -- "does interval x overlap anything
// already in room r?" -- an O(log) predecessor/successor probe, and it makes
// the targeted "ejection" move cheap: to force a high-weight interval into a
// room, we look up exactly the few placed intervals it collides with (a
// contiguous run in the ordered set), tentatively remove them, and decide with
// a simulated-annealing criterion on the resulting weight delta. Because the
// feasibility recheck is confined to the affected room's set, each SA step is
// near-O(log n), so we can take millions of steps inside the time budget.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    inline uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
    inline uint32_t u32() { return (uint32_t)(next() >> 32); }
    inline int randint(int n) { return (int)(u32() % (uint32_t)n); }   // [0,n)
    inline double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double t_start = now_sec();
    const double TIME_LIMIT = 1.9;   // seconds

    int n, K;
    if (!(cin >> n >> K)) return 0;

    vector<int> S(n), E(n), W(n);
    for (int i = 0; i < n; i++) cin >> S[i] >> E[i] >> W[i];

    // Guard against degenerate sizes so we always emit a feasible line per job.
    if (n <= 0) { return 0; }
    if (K <= 0) { for (int i = 0; i < n; i++) cout << -1 << "\n"; return 0; }

    // Per room, an ordered set of placed intervals keyed by (start, id). Storing
    // start lets us find, for any candidate, the placed intervals it overlaps as
    // a contiguous neighborhood: everything with start < cand.end whose end >
    // cand.start. We probe the predecessor (largest start <= cand.start) and walk
    // forward over successors while their start < cand.end.
    //
    // Each entry is (start, id). Lookups use end[]/start[] arrays by id.
    vector<set<pair<int,int>>> room(K);   // set of (start, id)
    vector<int> assign(n, -1);            // current room of each interval, -1 = rejected
    long long curWeight = 0;

    // ---- helper: do the placed intervals in room r conflict with interval x?
    //      If so, collect the conflicting ids into `hit`. Returns total weight of
    //      conflicts (so the caller can decide whether ejecting them is worth it).
    //
    // KEY INVARIANT: the intervals already in a room are pairwise non-overlapping,
    // hence sorted by start AND by end at the same time. So an interval p overlaps
    // x = [S,E) iff p.start < E and p.end > S, and:
    //   * Among predecessors (start <= S[x]) only the single closest one -- the
    //     one with the largest start -- can reach past S[x]; everything earlier
    //     ends even earlier (disjointness), so it cannot overlap. One probe.
    //   * Among successors (start in [S[x], E[x])) every one overlaps x, since its
    //     start lies strictly inside x. They form a contiguous run we walk.
    auto collectConflicts = [&](int r, int x, vector<int>& hit) -> long long {
        hit.clear();
        long long wsum = 0;
        const set<pair<int,int>>& st = room[r];
        auto it = st.lower_bound(make_pair(S[x], INT_MIN));
        // 1) the single predecessor (largest start < S[x]) -- check it only.
        if (it != st.begin()) {
            auto pit = prev(it);
            int pid = pit->second;
            if (E[pid] > S[x]) {               // reaches past S[x] -> overlaps
                hit.push_back(pid); wsum += W[pid];
            }
        }
        // 2) intervals whose start is in [S[x], E[x]) -- all overlap x.
        for (; it != st.end() && it->first < E[x]; ++it) {
            int pid = it->second;
            hit.push_back(pid); wsum += W[pid];
        }
        return wsum;
    };

    auto placeInterval = [&](int x, int r) {
        room[r].insert(make_pair(S[x], x));
        assign[x] = r;
        curWeight += W[x];
    };
    auto removeInterval = [&](int x) {
        int r = assign[x];
        room[r].erase(make_pair(S[x], x));
        assign[x] = -1;
        curWeight -= W[x];
    };

    // ---------------- construction: greedy by weight (descending) -------------
    // For each interval in descending weight, try to place it in some room with
    // no conflict. We try all K rooms (K is small) and place in the first that
    // is conflict-free; if none, leave rejected.
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){
        if (W[a] != W[b]) return W[a] > W[b];
        return (E[a]-S[a]) < (E[b]-S[b]);     // tie: shorter first (easier to pack)
    });

    vector<int> hit;
    for (int idx = 0; idx < n; idx++) {
        int x = order[idx];
        int chosen = -1;
        for (int r = 0; r < K; r++) {
            long long c = collectConflicts(r, x, hit);
            if (hit.empty()) { chosen = r; break; }
            (void)c;
        }
        if (chosen >= 0) placeInterval(x, chosen);
    }

    // best snapshot
    vector<int> bestAssign = assign;
    long long bestWeight = curWeight;

    // ---------------- SA with targeted ejection moves -------------------------
    Rng rng(0xC0FFEEULL ^ ((uint64_t)n << 20) ^ ((uint64_t)K << 8));

    double T0 = 200.0, T1 = 0.5;   // SA temperature schedule on weight units
    long long iter = 0;
    double temp = T0;

    // We keep a list of all interval ids to sample candidates from.
    while (true) {
        if ((iter & 1023) == 0) {
            double el = now_sec() - t_start;
            if (el > TIME_LIMIT) break;
            double frac = el / TIME_LIMIT;
            temp = T0 * pow(T1 / T0, frac);
        }
        iter++;

        int x = rng.randint(n);
        int r = rng.randint(K);

        if (assign[x] == r) {
            // Move: try ejecting x to a different room or rejecting it. With
            // prob, just attempt re-insertion elsewhere via the generic path
            // below by picking a different room.
            continue;
        }

        // Ejection move: force x into room r. Find the intervals x collides with
        // in room r; they are the ejection set. The new total weight would be
        //   cur - (x's current contribution) - (weight of ejected) + W[x].
        // x's current contribution is W[x] if it is already accepted, else 0.
        // Note: assign[x] == r was rejected above, so x is not in room r; ejecting
        // x's own old room and the conflicts in r are disjoint operations.
        long long ejW = collectConflicts(r, x, hit);
        int xCur = assign[x];
        long long delta = ((xCur == -1) ? 0 : -(long long)W[x]) - ejW + (long long)W[x];

        bool accept;
        if (delta >= 0) accept = true;
        else accept = (rng.uni() < exp((double)delta / max(1e-9, temp)));

        if (!accept) continue;

        // apply: remove x from its old room (if any), eject conflicts, place x.
        // copy hit because removeInterval mutates the set but not `hit`.
        if (xCur != -1) removeInterval(x);
        // re-collect conflicts in r (x removal can't affect room r unless xCur==r,
        // which is impossible here since assign[x]==r was handled above).
        // hit already holds the conflict ids for room r and x.
        for (int pid : hit) removeInterval(pid);
        placeInterval(x, r);

        if (curWeight > bestWeight) {
            bestWeight = curWeight;
            bestAssign = assign;
        }
    }

    // Make sure we report the best snapshot seen.
    // (curWeight may be below bestWeight after exploratory moves.)
    vector<int>& out = bestAssign;

    // ---- final safety: validate the snapshot is actually feasible. If, for any
    //      reason, it is not, fall back to the greedy-by-weight feasible build,
    //      which is guaranteed conflict-free by construction. This guarantees we
    //      never print an infeasible solution.
    {
        bool ok = true;
        vector<vector<pair<int,int>>> chk(K);
        for (int i = 0; i < n && ok; i++) {
            int r = out[i];
            if (r == -1) continue;
            if (r < 0 || r >= K) { ok = false; break; }
            chk[r].push_back(make_pair(S[i], E[i]));
        }
        if (ok) {
            for (int r = 0; r < K && ok; r++) {
                sort(chk[r].begin(), chk[r].end());
                for (size_t j = 1; j < chk[r].size(); j++) {
                    if (chk[r][j].first < chk[r][j-1].second) { ok = false; break; }
                }
            }
        }
        if (!ok) {
            // rebuild greedy-by-weight feasibly from scratch
            vector<int> fa(n, -1);
            vector<set<pair<int,int>>> rm(K);
            for (int idx = 0; idx < n; idx++) {
                int x = order[idx];
                for (int r = 0; r < K; r++) {
                    bool conflict = false;
                    auto& st = rm[r];
                    auto it = st.lower_bound(make_pair(S[x], -1));
                    if (it != st.begin()) {
                        auto pit = prev(it);
                        if (E[pit->second] > S[x]) conflict = true;
                    }
                    if (!conflict && it != st.end() && it->first < E[x]) conflict = true;
                    if (!conflict) { st.insert(make_pair(S[x], x)); fa[x] = r; break; }
                }
            }
            out = fa;
        }
    }

    // emit
    string buf;
    buf.reserve(n * 3);
    for (int i = 0; i < n; i++) {
        buf += to_string(out[i]);
        buf += '\n';
    }
    cout << buf;
    return 0;
}

// RCPSP (Resource-Constrained Project Scheduling) heuristic solver.
//
// Reads a project instance from stdin, writes one start time per task to stdout.
//
// INSTANCE (stdin):
//     n R
//     cap_1 ... cap_R
//     then n lines, task i (1-indexed):
//         dur  d_{i,1} ... d_{i,R}  p  pred_1 ... pred_p
//   Discrete time from 0. Resource k has constant capacity cap_k. Task i runs
//   for dur_i over [s_i, s_i+dur_i) consuming d_{i,k} of resource k. Precedence
//   is finish-to-start: s_i >= s_j + dur_j for every predecessor j of i.
//
// SOLUTION (stdout): n integers, the start time of each task in input order.
//   The objective is to MINIMIZE the makespan max_i (s_i + dur_i). An infeasible
//   schedule scores 0, so we only ever emit feasible schedules.
//
// METHOD (the innovation):
//   * Representation: a *priority list* (activity list) = a permutation of tasks
//     that is a topological order of the precedence DAG.
//   * Decoder: the SERIAL schedule-generation scheme (SGS). Tasks are scheduled
//     in list order; each task is placed at the earliest time >= its predecessor
//     finish at which every resource has room for its whole duration. Because we
//     only ever schedule a task after its predecessors (the list is topological)
//     and never exceed capacity, the SGS produces a precedence- AND
//     resource-feasible schedule *by construction* -- feasibility is free.
//   * Fast feasibility: a per-resource time-profile array prof[k][t] = units of
//     resource k already booked at time t. Finding the earliest feasible start
//     for a task scans candidate starts and, for each, checks the duration
//     window against every resource; booking a task updates the profile
//     incrementally. We grow the profile horizon lazily.
//   * Search: simulated annealing over the priority list. The move shifts one
//     task to a new position that is still a valid topological order (it must
//     stay after all its predecessors and before all its successors); the
//     decode-and-evaluate of a candidate list is O(n * horizon * R) but cheap in
//     practice, and we keep the best schedule ever decoded.
//   * Polish: forward-backward improvement ("double justification") -- decode
//     left-justified, then right-justify (schedule in reverse-finish order
//     against a mirrored profile), then left-justify again. Each justification
//     never increases the makespan and usually shrinks it; it is the standard
//     cheap RCPSP booster.

#include <bits/stdc++.h>
using namespace std;

static const double TIME_LIMIT = 1.85; // seconds

static int N, R;
static vector<int> cap;                 // [R]
static vector<int> dur;                 // [N]
static vector<vector<int>> dem;         // [N][R]
static vector<vector<int>> preds;       // [N]
static vector<vector<int>> succ;        // [N]
static vector<int> posInList;           // helper: position of task in a list

// ----------------------------------------------------------------- timing
static chrono::steady_clock::time_point T0;
static inline double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}

// ----------------------------------------------------------------- RNG
static uint64_t rngState = 88172645463325252ULL;
static inline uint64_t xrand() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return rngState;
}
static inline int randInt(int lo, int hi) { // inclusive
    return lo + (int)(xrand() % (uint64_t)(hi - lo + 1));
}
static inline double randDouble() {
    return (xrand() >> 11) * (1.0 / 9007199254740992.0);
}

// ----------------------------------------------------------------- profiles
// prof[k] is a difference-free running array of booked usage per time unit.
struct Profiles {
    int H;                       // current horizon length (>= any time we book)
    vector<vector<int>> prof;    // [R][H]
    void init(int hint) {
        H = max(8, hint);
        prof.assign(R, vector<int>(H, 0));
    }
    void clearTo(int hint) {
        if ((int)prof.empty()) { init(hint); return; }
        int need = max(8, hint);
        if (need > H) {
            H = need;
            for (int k = 0; k < R; k++) prof[k].assign(H, 0);
        } else {
            for (int k = 0; k < R; k++)
                fill(prof[k].begin(), prof[k].end(), 0);
        }
    }
    void ensure(int t) {
        if (t < H) return;
        int nH = H;
        while (nH <= t) nH = nH + nH / 2 + 8;
        for (int k = 0; k < R; k++) prof[k].resize(nH, 0);
        H = nH;
    }
};

// Find the earliest start >= t0 at which task i fits for its whole duration in
// the forward profile P, then (optionally) book it.
static int placeForward(Profiles &P, int i, int t0) {
    int d = dur[i];
    int t = t0;
    while (true) {
        P.ensure(t + d);
        bool ok = true;
        int advance = t + 1;               // default: try the next unit
        for (int k = 0; k < R && ok; k++) {
            int dd = dem[i][k];
            if (!dd) continue;
            int cp = cap[k];
            const int *row = P.prof[k].data();
            for (int tt = t; tt < t + d; tt++) {
                if (row[tt] + dd > cp) {
                    ok = false;
                    // any start <= tt overlaps the blocking unit, so the next
                    // possibly-feasible start is tt+1: skip a whole block.
                    advance = max(advance, tt + 1);
                    break;
                }
            }
        }
        if (ok) return t;
        t = advance;
    }
}

static inline void bookForward(Profiles &P, int i, int s) {
    int d = dur[i];
    P.ensure(s + d);
    for (int k = 0; k < R; k++) {
        int dd = dem[i][k];
        if (!dd) continue;
        int *row = P.prof[k].data();
        for (int tt = s; tt < s + d; tt++) row[tt] += dd;
    }
}

// ------------------------------------------------- serial SGS (left-justified)
// Decode a priority list (topological order) into start times; returns makespan.
static Profiles fwdP;
static int decodeSerial(const vector<int> &list, vector<int> &start) {
    int hint = 0; for (int i = 0; i < N; i++) hint += dur[i];
    fwdP.clearTo(hint + 1);
    int mk = 0;
    for (int idx = 0; idx < N; idx++) {
        int i = list[idx];
        int t0 = 0;
        for (int j : preds[i]) t0 = max(t0, start[j] + dur[j]);
        int s = placeForward(fwdP, i, t0);
        start[i] = s;
        bookForward(fwdP, i, s);
        mk = max(mk, s + dur[i]);
    }
    return mk;
}

// ----------------------------------------- right-justified decode (mirror SGS)
// Given a makespan T, schedule tasks in order of *decreasing* (start+dur) so
// each finishes as late as possible (its successors fixed), against a mirrored
// profile. Returns new start times that finish at <= T and a (possibly smaller)
// effective makespan after re-left-justifying.
static Profiles bwdP;
static int rightJustify(int T, vector<int> &start) {
    // order tasks by decreasing current finish time; ties by decreasing start.
    static vector<int> ord;
    ord.resize(N);
    for (int i = 0; i < N; i++) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        int fa = start[a] + dur[a], fb = start[b] + dur[b];
        if (fa != fb) return fa > fb;
        return start[a] > start[b];
    });
    // mirror time: place each task as late as possible. We work in mirrored time
    // tau = T - finish; equivalently use a backward profile indexed by mirrored
    // start. Simpler: compute the latest feasible finish given successors.
    bwdP.clearTo(T + 1);
    vector<int> fin(N, 0);
    for (int idx = 0; idx < N; idx++) {
        int i = ord[idx];
        int d = dur[i];
        // latest finish <= T and <= min successor start
        int hi = T;
        for (int j : succ[i]) hi = min(hi, fin[j] - dur[j]); // succ start = fin[j]-dur[j]
        // find latest start s in [0, hi-d] with room; scan downward using the
        // mirrored profile (index by start). To keep it simple and robust we scan
        // from hi-d downward.
        int s = hi - d;
        while (s >= 0) {
            bwdP.ensure(s + d);
            bool ok = true;
            for (int k = 0; k < R && ok; k++) {
                int dd = dem[i][k]; if (!dd) continue;
                int cp = cap[k]; const int *row = bwdP.prof[k].data();
                for (int tt = s; tt < s + d; tt++)
                    if (row[tt] + dd > cp) { ok = false; break; }
            }
            if (ok) break;
            s--;
        }
        if (s < 0) s = 0; // safety: should not happen, fall back
        bwdP.ensure(s + d);
        for (int k = 0; k < R; k++) {
            int dd = dem[i][k]; if (!dd) continue;
            int *row = bwdP.prof[k].data();
            for (int tt = s; tt < s + d; tt++) row[tt] += dd;
        }
        start[i] = s;
        fin[i] = s + d;
    }
    int mk = 0; for (int i = 0; i < N; i++) mk = max(mk, start[i] + dur[i]);
    return mk;
}

// Build a priority list (topological order) from current start times: sort by
// start time, breaking ties so predecessors precede successors (stable by index
// is not enough; we sort by (start, then a topo-rank)). We use a Kahn pass keyed
// by start time to guarantee a valid topological order.
static vector<int> listFromStarts(const vector<int> &start) {
    vector<int> indeg(N), list; list.reserve(N);
    for (int i = 0; i < N; i++) indeg[i] = (int)preds[i].size();
    // priority queue of ready tasks keyed by (start, index)
    auto cmp = [&](int a, int b) {
        if (start[a] != start[b]) return start[a] > start[b];
        return a > b;
    };
    priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < N; i++) if (indeg[i] == 0) pq.push(i);
    while (!pq.empty()) {
        int i = pq.top(); pq.pop();
        list.push_back(i);
        for (int j : succ[i]) if (--indeg[j] == 0) pq.push(j);
    }
    return list;
}

int main() {
    T0 = chrono::steady_clock::now();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> N >> R)) { return 0; }
    cap.resize(R);
    for (int k = 0; k < R; k++) cin >> cap[k];
    dur.assign(N, 0);
    dem.assign(N, vector<int>(R, 0));
    preds.assign(N, {});
    succ.assign(N, {});
    for (int i = 0; i < N; i++) {
        cin >> dur[i];
        for (int k = 0; k < R; k++) cin >> dem[i][k];
        int p; cin >> p;
        preds[i].resize(p);
        for (int t = 0; t < p; t++) { int j; cin >> j; preds[i][t] = j - 1; }
    }
    for (int i = 0; i < N; i++)
        for (int j : preds[i]) succ[j].push_back(i);

    if (N == 0) { return 0; }

    // --- initial priority list: a topological order (Kahn by index). ----------
    vector<int> list; list.reserve(N);
    {
        vector<int> indeg(N);
        for (int i = 0; i < N; i++) indeg[i] = (int)preds[i].size();
        // earliest-index-first ready selection (the trivial baseline list)
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int i = 0; i < N; i++) if (indeg[i] == 0) pq.push(i);
        while (!pq.empty()) {
            int i = pq.top(); pq.pop();
            list.push_back(i);
            for (int j : succ[i]) if (--indeg[j] == 0) pq.push(j);
        }
    }
    posInList.assign(N, 0);

    vector<int> curStart(N, 0), bestStart(N, 0), tmpStart(N, 0);
    int curMk = decodeSerial(list, curStart);

    // double justification on the initial schedule
    {
        tmpStart = curStart;
        int m1 = rightJustify(curMk, tmpStart);
        vector<int> l2 = listFromStarts(tmpStart);
        int m2 = decodeSerial(l2, tmpStart);
        if (m2 <= curMk) { curMk = m2; curStart = tmpStart; list = l2; }
        (void)m1;
    }
    int bestMk = curMk; bestStart = curStart;
    vector<int> bestList = list;

    // --- simulated annealing over the priority list. -------------------------
    // Move: pick a task, move it to a new feasible position in the list (between
    // the latest of its predecessors and the earliest of its successors).
    double T = 0.0, Tend = 0.0;
    {
        // crude temperature scale from makespan magnitude
        T = max(1.0, curMk * 0.06);
        Tend = max(0.05, curMk * 0.001);
    }
    long iter = 0;
    int sinceImprove = 0;
    while (true) {
        if ((iter & 255) == 0) {
            double el = elapsed();
            if (el > TIME_LIMIT) break;
            double frac = el / TIME_LIMIT;
            T = max(Tend, (curMk * 0.06) * pow(Tend / (curMk * 0.06 + 1e-9), frac));
        }
        iter++;

        // rebuild position index for the current list
        for (int idx = 0; idx < N; idx++) posInList[list[idx]] = idx;

        int task = randInt(0, N - 1);
        int oldPos = posInList[task];
        // feasible window [lo, hi] for task's new position
        int lo = 0, hi = N - 1;
        for (int j : preds[task]) lo = max(lo, posInList[j] + 1);
        for (int j : succ[task]) hi = min(hi, posInList[j] - 1);
        if (hi < lo) continue;
        int newPos = randInt(lo, hi);
        if (newPos == oldPos) continue;

        // construct candidate list by moving task from oldPos to newPos.
        // newPos is a target index in the ORIGINAL list. After erasing oldPos,
        // every element after oldPos shifts left by one, so the insertion index
        // is newPos when newPos < oldPos and newPos-1 when newPos > oldPos. The
        // result is a list in which `task` ends up at index newPos and the rest
        // keep their relative order -- still a valid topological order because
        // newPos lies strictly between every predecessor and successor position.
        static vector<int> cand; cand = list;
        cand.erase(cand.begin() + oldPos);
        int insAt = (newPos > oldPos) ? newPos - 1 : newPos;
        cand.insert(cand.begin() + insAt, task);

        int candMk = decodeSerial(cand, tmpStart);

        int delta = candMk - curMk;
        bool accept = (delta <= 0) || (randDouble() < exp(-delta / T));
        if (accept) {
            curMk = candMk; curStart = tmpStart; list = cand;
            if (candMk < bestMk) {
                bestMk = candMk; bestStart = tmpStart; bestList = cand;
                sinceImprove = 0;
            }
        }

        // periodic double-justification polish from the current best
        if (++sinceImprove >= 4000) {
            sinceImprove = 0;
            tmpStart = bestStart;
            int m1 = rightJustify(bestMk, tmpStart);
            vector<int> l2 = listFromStarts(tmpStart);
            int m2 = decodeSerial(l2, tmpStart);
            if (m2 < bestMk) {
                bestMk = m2; bestStart = tmpStart; bestList = l2;
                // also adopt as current to keep searching from the improvement
                curMk = m2; curStart = tmpStart; list = l2;
            }
            (void)m1;
        }
    }

    // final double justification on the best
    {
        tmpStart = bestStart;
        int m1 = rightJustify(bestMk, tmpStart);
        vector<int> l2 = listFromStarts(tmpStart);
        int m2 = decodeSerial(l2, tmpStart);
        if (m2 < bestMk) { bestMk = m2; bestStart = tmpStart; }
        (void)m1;
    }

    // emit the best feasible start times
    {
        string out; out.reserve(N * 7);
        for (int i = 0; i < N; i++) {
            out += to_string(bestStart[i]);
            out += (i + 1 < N) ? ' ' : '\n';
        }
        fputs(out.c_str(), stdout);
    }
    return 0;
}

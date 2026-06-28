// ale-v2-02 : Rectangle Strip Packing (minimize used strip height).
//
// Read: "W N", then N rectangles "w_i h_i" (no rotation, w_i <= W).
// Write: N lines "x_i y_i", the bottom-left corner of rectangle i (input order),
//        a non-overlapping packing inside the strip of width W; minimize the
//        used height H = max_i (y_i + h_i).
//
// Method (the innovation): bottom-left-fill (BLF) decoding driven by an
// INSERTION ORDER, with a SKYLINE so each placement is incremental and cheap;
// simulated annealing searches over the permutation. BLF places each rectangle
// at the lowest feasible y, then leftmost x, by sliding a width-w window over
// the skyline. The skyline is the upper contour of what is already packed, so a
// placement only needs to scan/merge the few segments it spans -- not re-pack
// from scratch. SA proposes swap / insert / reverse moves on the order and
// keeps the order whose BLF height is smallest (Metropolis acceptance).
//
// Always feasible: BLF of ANY permutation yields a valid non-overlapping
// packing inside the strip (every rectangle has w_i <= W), so we can never emit
// an invalid solution.

#include <bits/stdc++.h>
using namespace std;

static int W, N;
static vector<int> rw, rh;            // rectangle widths / heights (input order)

// ---- skyline ----------------------------------------------------------------
// The skyline is a sequence of segments covering [0, W): each segment is a flat
// top at height `h` over [x, x_next). Stored as parallel arrays of breakpoints.
struct Skyline {
    // segs: sorted breakpoints; seg i covers [xs[i], xs[i+1]) at height hs[i].
    vector<int> xs;   // size m+1, xs[0]=0, xs[m]=W
    vector<int> hs;   // size m
    void reset() {
        xs.assign(2, 0);
        xs[0] = 0; xs[1] = W;
        hs.assign(1, 0);
    }
    // Lowest y at which a width-w rectangle whose left edge is at x can sit:
    // the max skyline height over [x, x+w).
    // Place a rectangle of width w using bottom-left-fill. Returns (x, y).
    // Scans candidate left positions (segment starts); for each, computes the
    // span height as the max segment height over [x, x+w); picks lowest y then
    // leftmost x. Then mutates the skyline to lay the rectangle on top.
};

// Compute the BLF placement of a rectangle of width w on the current skyline.
// Returns chosen (x, y). Guaranteed to find a spot because w <= W.
static inline void blf_place(const Skyline& sk, int w, int& bx, int& by) {
    const vector<int>& xs = sk.xs;
    const vector<int>& hs = sk.hs;
    int m = (int)hs.size();
    bx = 0; by = INT_MAX;
    // Candidate left edges: each segment start such that x + w <= W.
    for (int i = 0; i < m; ++i) {
        int x = xs[i];
        if (x + w > W) break;          // xs is increasing; no later start fits
        // height = max hs[k] over segments k that intersect [x, x+w)
        int xr = x + w;
        int hmax = 0;
        for (int k = i; k < m && xs[k] < xr; ++k) {
            if (xs[k + 1] <= x) continue;   // shouldn't happen given k>=i
            if (hs[k] > hmax) hmax = hs[k];
        }
        if (hmax < by) { by = hmax; bx = x; }
    }
    if (by == INT_MAX) { by = 0; bx = 0; }  // safety (never expected)
}

// Lay a rectangle [x, x+w) x [y, y+h) onto the skyline: raise the covered span
// to y+h, keeping the breakpoint arrays normalized (merge equal-height runs).
static inline void blf_commit(Skyline& sk, int x, int y, int w, int h) {
    int top = y + h;
    int xr = x + w;
    vector<int>& xs = sk.xs;
    vector<int>& hs = sk.hs;
    vector<int> nxs, nhs;
    nxs.reserve(xs.size() + 2);
    nhs.reserve(hs.size() + 2);
    int m = (int)hs.size();
    for (int i = 0; i < m; ++i) {
        int a = xs[i], b = xs[i + 1], hh = hs[i];
        // Split segment [a,b) against [x, xr): the overlap becomes height `top`.
        int oa = max(a, x), ob = min(b, xr);
        if (oa >= ob) {
            // no overlap -> keep as is
            nxs.push_back(a); nhs.push_back(hh);
            continue;
        }
        // left remnant [a, oa)
        if (a < oa) { nxs.push_back(a); nhs.push_back(hh); }
        // raised middle [oa, ob)
        nxs.push_back(oa); nhs.push_back(top);
        // right remnant [ob, b)
        if (ob < b) { nxs.push_back(ob); nhs.push_back(hh); }
    }
    // Rebuild breakpoint array, merging adjacent equal heights.
    vector<int> oxs, ohs;
    oxs.reserve(nxs.size() + 1);
    ohs.reserve(nhs.size());
    for (size_t i = 0; i < nhs.size(); ++i) {
        if (!ohs.empty() && ohs.back() == nhs[i]) {
            // merge: extend previous segment (drop this start)
            continue;
        }
        oxs.push_back(nxs[i]);
        ohs.push_back(nhs[i]);
    }
    oxs.push_back(W);
    sk.xs = std::move(oxs);
    sk.hs = std::move(ohs);
}

// Decode a permutation `order` into placements via BLF; fills px,py and returns
// the used height H. O(N * segments) per decode.
static Skyline gsk;  // reused scratch skyline
static int decode(const vector<int>& order, vector<int>& px, vector<int>& py) {
    gsk.reset();
    int H = 0;
    for (int idx : order) {
        int w = rw[idx], h = rh[idx];
        int x, y;
        blf_place(gsk, w, x, y);
        blf_commit(gsk, x, y, w, h);
        px[idx] = x; py[idx] = y;
        H = max(H, y + h);
    }
    return H;
}

int main() {
    if (scanf("%d %d", &W, &N) != 2) return 0;
    rw.resize(N); rh.resize(N);
    for (int i = 0; i < N; ++i) scanf("%d %d", &rw[i], &rh[i]);

    if (N == 0) { return 0; }

    // Initial order: tallest-first is a strong BLF seed for strip packing
    // (decreasing height tends to lay a stable base).
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [](int a, int b){
        if (rh[a] != rh[b]) return rh[a] > rh[b];
        return rw[a] > rw[b];
    });

    vector<int> px(N), py(N), bpx(N), bpy(N);
    int H = decode(order, px, py);

    vector<int> best = order;
    int bestH = H;
    bpx = px; bpy = py;

    // ---- simulated annealing over the insertion order ----
    std::mt19937 rng(987654321u);
    auto t0 = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.8;     // seconds
    double T0 = 60.0, T1 = 0.5;
    vector<int> cur = order;
    int curH = H;
    long iter = 0;
    double T = T0;
    while (true) {
        if ((iter & 255) == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            if (el > TIME_LIMIT) break;
            double frac = el / TIME_LIMIT;
            T = T0 * pow(T1 / T0, frac);
        }
        ++iter;
        // propose a move on a copy
        vector<int> cand = cur;
        int mv = rng() % 3;
        int i = rng() % N, j = rng() % N;
        if (i == j) j = (j + 1) % N;
        if (mv == 0) {            // swap
            std::swap(cand[i], cand[j]);
        } else if (mv == 1) {     // move element i to position j (insert)
            int v = cand[i];
            cand.erase(cand.begin() + i);
            cand.insert(cand.begin() + j, v);
        } else {                  // reverse segment [min,max]
            int a = min(i, j), b = max(i, j);
            reverse(cand.begin() + a, cand.begin() + b + 1);
        }
        int candH = decode(cand, px, py);
        int dH = candH - curH;
        bool accept;
        if (dH <= 0) accept = true;
        else {
            double p = exp(-(double)dH / max(1e-9, T));
            accept = ((double)(rng() & 0xffffff) / (double)0x1000000) < p;
        }
        if (accept) {
            cur.swap(cand);
            curH = candH;
            if (curH < bestH) {
                bestH = curH;
                best = cur;
                bpx = px; bpy = py;   // px/py hold the decode of `cand`==`cur`
            }
        }
    }

    // Re-decode best to be certain px/py match the emitted order.
    decode(best, bpx, bpy);

    // Emit placements in INPUT order.
    string out;
    out.reserve(N * 12);
    char buf[32];
    for (int i = 0; i < N; ++i) {
        int len = snprintf(buf, sizeof(buf), "%d %d\n", bpx[i], bpy[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}

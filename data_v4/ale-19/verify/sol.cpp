// 2D Rectangle Strip Packing -- heuristic solver.
//
// Objective: place n axis-aligned rectangles into a vertical strip of fixed
// integer width W and unbounded height, without overlap, so as to MINIMIZE the
// used height = max over placed rectangles of (y + height). R is a global flag:
// R == 1 lets a rectangle be rotated 90 degrees, R == 0 fixes the orientation.
// We read the instance from stdin and write, to stdout, n lines
//     x_i y_i r_i
// the bottom-left corner and rotation bit of rectangle i (input order).
//
// Method (the innovation): SKYLINE bottom-left placement driven by a PERMUTATION
// plus a per-rectangle ROTATE bit, optimized by SIMULATED ANNEALING over the
// permutation (and the rotate bits).
//   * The strip's frontier is stored as a SKYLINE: a list of horizontal segments
//     (x, width, y) that partition [0, W]. To place a rectangle of placed width
//     pw, the only sensible x-positions are the left edges of skyline segments.
//     For a candidate x the rectangle must rest on top of the HIGHEST skyline
//     segment overlapping the span [x, x+pw); that is its bottom y. We pick the
//     (x, orientation) minimizing the resulting top y+ph, with a bottom-left /
//     least-waste tie-break, then RAISE the skyline over [x, x+pw) to y+ph
//     (merging equal-height neighbours). Each placement is O(#segments) and a
//     full construction (replay of a permutation) is O(n * #segments) -- tiny --
//     which is exactly what makes the SA candidate evaluation cheap.
//   * The decision variables are the INSERTION ORDER and, when R == 1, the
//     per-item rotation. SA swaps two positions / moves an item / flips a rotate
//     bit, REBUILDS the packing by replay, and accepts by the Metropolis rule on
//     the height. Skyline placement is overlap-free and in-strip BY
//     CONSTRUCTION, so every replay -- hence any early stop -- prints a feasible
//     solution. We keep the best order seen and emit its packing.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N, W, R;
vector<int> RW, RH;               // requested rectangle dimensions (native)

// A skyline segment covers [x, x+width) at height y. Segments partition [0, W),
// kept left-to-right, contiguous, with no two equal-height neighbours.
struct Seg { int x, w, y; };

// Place rectangle with placed dims (pw, ph) onto skyline `sky` using bottom-left
// with least-waste tie-break. Returns the chosen bottom-left (px, py) and the
// achieved top (py+ph); updates `sky`. `pw` is assumed <= W (guaranteed by the
// orientation choice). Returns the top y after placement.
static inline int place_on_skyline(vector<Seg>& sky, int pw, int ph,
                                   int& out_x, int& out_y) {
    int best_i = -1, best_y = INT_MAX, best_x = 0, best_waste = INT_MAX;
    int S = (int)sky.size();
    for (int i = 0; i < S; ++i) {
        int x = sky[i].x;
        if (x + pw > W) break;           // segments sorted by x; no room further right
        // bottom y = max skyline height over [x, x+pw)
        int span = pw, j = i, ytop = 0, waste = 0;
        while (span > 0 && j < S) {
            if (sky[j].y > ytop) ytop = sky[j].y;
            span -= sky[j].w;
            ++j;
        }
        if (span > 0) break;             // ran off the right end: cannot fit here
        // recompute wasted area below the rectangle (lower => tighter fit)
        span = pw; j = i; waste = 0;
        while (span > 0 && j < S) {
            int cw = min(sky[j].w, span);
            waste += (ytop - sky[j].y) * cw;
            span -= sky[j].w;
            ++j;
        }
        int top = ytop + ph;
        if (top < best_y || (top == best_y && (waste < best_waste ||
            (waste == best_waste && x < best_x)))) {
            best_y = top; best_i = i; best_x = x; best_waste = waste;
        }
    }
    if (best_i < 0) {                    // should not happen (pw <= W) but be safe
        // place at the global skyline max, far left
        int ymax = 0; for (auto& s : sky) ymax = max(ymax, s.y);
        out_x = 0; out_y = ymax;
        return ymax + ph;
    }
    int px = best_x;
    int py = best_y - ph;
    out_x = px; out_y = py;
    // Raise the skyline over [px, px+pw) to py+ph. Rebuild affected segments.
    int newtop = py + ph;
    vector<Seg> ns;
    ns.reserve(sky.size() + 2);
    int xr = px + pw;
    for (int i = 0; i < (int)sky.size(); ++i) {
        int sx = sky[i].x, sw = sky[i].w, sy = sky[i].y;
        int ex = sx + sw;
        if (ex <= px || sx >= xr) {      // untouched
            ns.push_back(sky[i]);
            continue;
        }
        // split into: left part [sx,px), raised part [max(sx,px),min(ex,xr)), right [xr,ex)
        if (sx < px) ns.push_back({sx, px - sx, sy});
        int rs = max(sx, px), re = min(ex, xr);
        if (re > rs) ns.push_back({rs, re - rs, newtop});
        if (ex > xr) ns.push_back({xr, ex - xr, sy});
    }
    // merge equal-height neighbours
    vector<Seg> merged;
    merged.reserve(ns.size());
    for (auto& s : ns) {
        if (!merged.empty() && merged.back().y == s.y &&
            merged.back().x + merged.back().w == s.x) {
            merged.back().w += s.w;
        } else {
            merged.push_back(s);
        }
    }
    sky.swap(merged);
    return newtop;
}

// Replay an insertion order (+ rotation bits) into a skyline packing.
// Fills px,py,pr (placement of each rectangle, by input index) when `out` set.
// Returns the used height (max top).
static int replay(const vector<int>& order, const vector<uint8_t>& rot,
                  vector<int>* px, vector<int>* py, vector<uint8_t>* pr) {
    vector<Seg> sky;
    sky.push_back({0, W, 0});
    int height = 0;
    for (int k = 0; k < N; ++k) {
        int idx = order[k];
        int w = RW[idx], h = RH[idx];
        int rbit = rot[idx];
        int pw = rbit ? h : w;
        int ph = rbit ? w : h;
        // safety: if this orientation does not fit the strip width, force the
        // other one (only possible when R==1 and one orientation is too wide).
        if (pw > W) { rbit ^= 1; swap(pw, ph); }
        int x, y;
        int top = place_on_skyline(sky, pw, ph, x, y);
        height = max(height, top);
        if (px) {
            (*px)[idx] = x; (*py)[idx] = y; (*pr)[idx] = (uint8_t)rbit;
        }
    }
    return height;
}

int main() {
    if (!(cin >> N >> W >> R)) return 0;
    RW.resize(N); RH.resize(N);
    for (int i = 0; i < N; ++i) cin >> RW[i] >> RH[i];

    if (N == 0) { return 0; }

    Rng rng(0x12345678ULL ^ ((uint64_t)N << 32) ^ ((uint64_t)W << 8) ^ (uint64_t)R);

    // ---- Initial order: decreasing height (a strong strip-packing seed). ----
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    // rotation seed: when allowed, orient each so the SHORTER side is the width
    // that fits (tends to make flatter pieces -> shorter strip); else keep.
    vector<uint8_t> rot(N, 0);
    for (int i = 0; i < N; ++i) {
        if (R == 1) {
            int w = RW[i], h = RH[i];
            bool fit0 = (w <= W), fit1 = (h <= W);
            // prefer the orientation with the larger width (wider, flatter), if it fits
            if (fit0 && fit1) rot[i] = (w >= h) ? 0 : 1;
            else if (fit1 && !fit0) rot[i] = 1;
            else rot[i] = 0;
        }
    }
    auto placed_h = [&](int i)->int { return rot[i] ? RW[i] : RH[i]; };
    sort(order.begin(), order.end(), [&](int a, int b) {
        int ha = placed_h(a), hb = placed_h(b);
        if (ha != hb) return ha > hb;
        return a < b;
    });

    vector<int> bpx(N), bpy(N); vector<uint8_t> bpr(N);
    int cur = replay(order, rot, &bpx, &bpy, &bpr);
    int best = cur;
    vector<int> best_order = order;
    vector<uint8_t> best_rot = rot;
    vector<int> best_px = bpx, best_py = bpy; vector<uint8_t> best_pr = bpr;

    // ---- Simulated annealing over the permutation (and rotate bits). ----
    const double TL = 1.8;             // seconds
    double t0 = now_sec();
    double T0 = max(4.0, best * 0.06), T1 = 0.4;
    long long iter = 0;
    vector<int> tpx(N), tpy(N); vector<uint8_t> tpr(N);

    while (true) {
        if ((iter & 255) == 0) {
            double el = now_sec() - t0;
            if (el > TL) break;
        }
        ++iter;
        double frac = (now_sec() - t0) / TL;
        if (frac > 1.0) frac = 1.0;
        double T = T0 * pow(T1 / T0, frac);

        // pick a move
        int mv = rng.nextu(R == 1 ? 3 : 2);
        int a = rng.nextu(N), b = rng.nextu(N);
        uint8_t saved_rot = 0; int saved_idx = -1;
        int type;
        if (mv == 2) {
            // flip a rotate bit (R==1 only)
            saved_idx = order[a];
            saved_rot = rot[saved_idx];
            rot[saved_idx] ^= 1;
            type = 2;
        } else if (mv == 0) {
            // swap two positions
            if (a == b) { b = (b + 1) % N; }
            swap(order[a], order[b]);
            type = 0;
        } else {
            // move element from a to position b (rotate the in-between block)
            if (a == b) { b = (b + 1) % N; }
            type = 1;
        }

        int cand;
        if (type == 1) {
            // build the moved order on the fly into a scratch and replay
            // (cheap enough at this n); apply, replay, then we can undo by symmetry
            int e = order[a];
            // erase a, insert before b
            // do it in-place on `order`
            // (store a copy of the touched range minimal: simplest is splice)
            int from = a, to = b;
            if (from < to) {
                for (int i = from; i < to; ++i) order[i] = order[i + 1];
                order[to] = e;
            } else {
                for (int i = from; i > to; --i) order[i] = order[i - 1];
                order[to] = e;
            }
            cand = replay(order, rot, &tpx, &tpy, &tpr);
            // undo for potential reject is handled below by re-splicing
            if (cand <= cur || rng.nextd() < exp((cur - cand) / max(1e-9, T))) {
                cur = cand;
                if (cand < best) {
                    best = cand; best_order = order; best_rot = rot;
                    best_px = tpx; best_py = tpy; best_pr = tpr;
                }
            } else {
                // undo the splice
                if (from < to) {
                    for (int i = to; i > from; --i) order[i] = order[i - 1];
                    order[from] = e;
                } else {
                    for (int i = to; i < from; ++i) order[i] = order[i + 1];
                    order[from] = e;
                }
            }
            continue;
        }

        cand = replay(order, rot, &tpx, &tpy, &tpr);
        bool accept = (cand <= cur) || (rng.nextd() < exp((cur - cand) / max(1e-9, T)));
        if (accept) {
            cur = cand;
            if (cand < best) {
                best = cand; best_order = order; best_rot = rot;
                best_px = tpx; best_py = tpy; best_pr = tpr;
            }
        } else {
            // undo
            if (type == 0) swap(order[a], order[b]);
            else if (type == 2) rot[saved_idx] = saved_rot;
        }
    }

    // ---- Emit the best packing (input order). ----
    // best_px/py/pr are indexed by input index.
    string out;
    out.reserve(N * 12);
    char buf[64];
    for (int i = 0; i < N; ++i) {
        int len = snprintf(buf, sizeof(buf), "%d %d %d\n",
                           best_px[i], best_py[i], (int)best_pr[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}

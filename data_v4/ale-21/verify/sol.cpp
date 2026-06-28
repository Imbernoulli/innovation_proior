// Guillotine Cutting Stock -- heuristic solver.
//
// Objective: place n requested rectangles onto W x H sheets (axis-aligned,
// optional 90-degree rotation), every per-sheet layout realizable by guillotine
// (edge-to-edge) cuts, so as to MINIMIZE
//     cost = (#sheets used) * W * H - (placed area) + P * (unplaced area)
// with P = 3. We read the instance from stdin and write, to stdout,
//     m
//     idx sheet x y rot         (m lines)
// for the m placed rectangles (omitted == unplaced).
//
// Method (the innovation):
//   * Encode feasibility into the CONSTRUCTION. Every sheet is a set of free
//     rectangles (a guillotine free-rectangle pool). To place a rectangle we
//     pick, over all sheets' free rectangles, a BEST-FIT free rectangle it fits
//     in (smallest leftover area, i.e. best-area-fit), drop it at that free
//     rect's bottom-left corner, and GUILLOTINE-SPLIT the leftover L-shape into
//     two rectangles (a full edge-to-edge cut). Because each placement is "drop
//     into a free rectangle + one guillotine split", the resulting layout is
//     guillotine-legal and overlap-free BY CONSTRUCTION -- we never have to
//     repair feasibility.
//   * The only decision left is the INSERTION ORDER (and per-item rotation). A
//     full construction (replay) costs O(n * #free-rects) which is tiny here, so
//     we run SIMULATED ANNEALING over the order: a move swaps two positions or
//     flips an item's rotation, we REBUILD the packing by replay, and accept by
//     the Metropolis rule on the cost above. The current best order's packing is
//     always a valid guillotine layout, so any early stop still prints a feasible
//     solution.
//   * The construction also greedily prefers to fill EXISTING sheets (only opens
//     a new sheet when no current free rectangle can host the item), which is
//     what drives the sheet count -- the dominant cost term -- down.
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

int N, W, H;
vector<int> RW, RH;          // requested rectangle dimensions
long long TOTAL_AREA = 0;
const long long P = 3;       // unplaced-area penalty (must match score.py)

struct Placement { int idx, sheet, x, y, rot; };

struct FreeRect { int x, y, w, h, sheet; };

// Replay a given order (+rotation bits) into a guillotine packing.
// Returns the cost and (optionally) fills `out` with the placements.
long long replay(const vector<int>& order, const vector<uint8_t>& rot,
                 vector<Placement>* out) {
    // free rectangles across all currently opened sheets
    static vector<FreeRect> freeR;
    freeR.clear();
    int sheets = 0;
    long long placedArea = 0;
    if (out) out->clear();

    auto openSheet = [&]() {
        FreeRect fr{0, 0, W, H, sheets};
        freeR.push_back(fr);
        sheets++;
    };

    for (size_t t = 0; t < order.size(); t++) {
        int id = order[t];
        int curRot = rot[id] ? 1 : 0;
        int w = curRot ? RH[id] : RW[id];
        int h = curRot ? RW[id] : RH[id];
        // If this orientation does not fit a fresh sheet, try the other one; if
        // neither fits, the rectangle is genuinely unplaceable and is skipped.
        if (w > W || h > H) {
            curRot ^= 1;
            std::swap(w, h);
            if (w > W || h > H) continue;  // truly cannot be placed in any orientation
        }

        // BEST-AREA-FIT: choose the free rectangle (already opened) with the
        // smallest leftover area that still fits w x h.
        int best = -1;
        long long bestLeft = LLONG_MAX;
        for (size_t i = 0; i < freeR.size(); i++) {
            const FreeRect& f = freeR[i];
            if (f.w >= w && f.h >= h) {
                long long left = (long long)f.w * f.h - (long long)w * h;
                if (left < bestLeft) { bestLeft = left; best = (int)i; }
            }
        }
        if (best < 0) {
            // no current free rectangle hosts it: open a new sheet.
            openSheet();
            best = (int)freeR.size() - 1;
            const FreeRect& f = freeR[best];
            if (!(f.w >= w && f.h >= h)) { continue; } // safety (shouldn't happen)
        }

        FreeRect f = freeR[best];
        // remove the consumed free rectangle (swap-pop)
        freeR[best] = freeR.back();
        freeR.pop_back();

        // place at (f.x, f.y); guillotine-split the leftover L into two rects.
        if (out) out->push_back(Placement{id, f.sheet, f.x, f.y, curRot});
        placedArea += (long long)w * h;

        int rightW = f.w - w;   // strip to the right of the placed rect
        int topH = f.h - h;     // strip above the placed rect
        // Choose the split that yields the more "useful" (larger-min-dim) pieces:
        // Shorter-Axis-Split heuristic -- split along the shorter leftover so the
        // bigger leftover piece stays whole (a standard guillotine split rule).
        bool splitHorizontalFirst = (rightW <= topH);
        if (splitHorizontalFirst) {
            // right piece spans the placed rect's height only; top piece spans full width
            if (rightW > 0) freeR.push_back(FreeRect{f.x + w, f.y, rightW, h, f.sheet});
            if (topH > 0)   freeR.push_back(FreeRect{f.x, f.y + h, f.w, topH, f.sheet});
        } else {
            // top piece spans the placed rect's width only; right piece spans full height
            if (topH > 0)   freeR.push_back(FreeRect{f.x, f.y + h, w, topH, f.sheet});
            if (rightW > 0) freeR.push_back(FreeRect{f.x + w, f.y, rightW, f.h, f.sheet});
        }
    }

    long long unplaced = TOTAL_AREA - placedArea;
    long long cost = (long long)sheets * W * H - placedArea + P * unplaced;
    return cost;
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;

    if (scanf("%d %d %d", &N, &W, &H) != 3) return 0;
    if (N <= 0) { printf("0\n"); return 0; }
    RW.resize(N); RH.resize(N);
    for (int i = 0; i < N; i++) {
        if (scanf("%d %d", &RW[i], &RH[i]) != 2) { RW[i] = 1; RH[i] = 1; }
        TOTAL_AREA += (long long)RW[i] * RH[i];
    }

    Rng rng(0xC0FFEEull ^ ((uint64_t)N << 32) ^ ((uint64_t)W << 16) ^ (uint64_t)H);

    // Initial order: largest-area first (a strong constructive start for packing),
    // with rotation set so the longer side is vertical (helps shelf-like growth).
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [](int a, int b) {
        long long aa = (long long)RW[a] * RH[a];
        long long bb = (long long)RW[b] * RH[b];
        if (aa != bb) return aa > bb;
        return a < b;
    });
    vector<uint8_t> rot(N, 0);
    for (int i = 0; i < N; i++) {
        // prefer the orientation that fits and is "tall" (h >= w) when possible
        int w = RW[i], h = RH[i];
        bool fit0 = (w <= W && h <= H);
        bool fit1 = (h <= W && w <= H);
        if (fit0 && (!fit1 || h >= w)) rot[i] = 0;
        else if (fit1) rot[i] = 1;
        else rot[i] = 0;
    }

    vector<int> curOrder = order;
    vector<uint8_t> curRot = rot;
    long long curCost = replay(curOrder, curRot, nullptr);

    vector<int> bestOrder = curOrder;
    vector<uint8_t> bestRot = curRot;
    long long bestCost = curCost;

    // Simulated annealing over (order, rotation). Cost scale ~ W*H per sheet.
    double Tstart = (double)W * H * 0.5 + 1.0;
    double Tend = 1.0;
    long long iters = 0;
    while (true) {
        if ((iters & 255) == 0) {
            double el = now_sec() - T0;
            if (el > TIME_LIMIT) break;
        }
        iters++;
        double frac = std::min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double Temp = Tstart * pow(Tend / Tstart, frac);

        // propose a move on a scratch copy
        vector<int> nxtOrder = curOrder;
        vector<uint8_t> nxtRot = curRot;
        int mv = rng.nextu(3);
        if (mv == 0 && N >= 2) {
            int i = rng.nextu(N), j = rng.nextu(N);
            std::swap(nxtOrder[i], nxtOrder[j]);
        } else if (mv == 1 && N >= 2) {
            // move one element to another position (shift)
            int i = rng.nextu(N), j = rng.nextu(N);
            int v = nxtOrder[i];
            nxtOrder.erase(nxtOrder.begin() + i);
            nxtOrder.insert(nxtOrder.begin() + j, v);
        } else {
            int i = rng.nextu(N);
            int id = curOrder[i];
            // only flip if the flipped orientation also fits a sheet
            int w = RH[id], h = RW[id];
            if (w <= W && h <= H) nxtRot[id] ^= 1;
            else { int k = rng.nextu(N); std::swap(nxtOrder[k], nxtOrder[(k + 1) % N]); }
        }

        long long nxtCost = replay(nxtOrder, nxtRot, nullptr);
        long long d = nxtCost - curCost;
        if (d <= 0 || rng.nextd() < exp(-(double)d / Temp)) {
            curOrder.swap(nxtOrder);
            curRot.swap(nxtRot);
            curCost = nxtCost;
            if (curCost < bestCost) {
                bestCost = curCost;
                bestOrder = curOrder;
                bestRot = curRot;
            }
        }
    }

    // Emit the best packing.
    vector<Placement> placements;
    replay(bestOrder, bestRot, &placements);

    string buf;
    buf.reserve(placements.size() * 16 + 16);
    buf += to_string(placements.size());
    buf += '\n';
    for (const auto& p : placements) {
        buf += to_string(p.idx); buf += ' ';
        buf += to_string(p.sheet); buf += ' ';
        buf += to_string(p.x); buf += ' ';
        buf += to_string(p.y); buf += ' ';
        buf += to_string(p.rot); buf += '\n';
    }
    fputs(buf.c_str(), stdout);
    return 0;
}

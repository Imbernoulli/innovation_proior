#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Retensioning the Ground-Braced Mast Array".
//
// Input:  n s c ; target[0..n-1] ; s strut lines (u v d, x_v-x_u=d exactly) ;
//         c cable lines (u v k). Exactly one endpoint of every cable is a FREE
//         (non-strut-grounded) node; the other is grounded. A free node's
//         position is the argmin over x of
//            e(x) = sum_{incident cable c} 0.5 * k_c * max(0, |x - o_c| - r_c)^2
//         (a tension-only spring to grounded point o_c with rest length r_c,
//         the solver's output) -- a strictly convex, piecewise-quadratic energy,
//         so this is the unique physical equilibrium of that mast tip, found by
//         bisecting its monotone derivative.
//
// Output: c non-negative reals (rest lengths), one per cable, input order.
//
// Objective (MIN):
//   F = sum_free_node (x* - target)^2                         [shape error]
//     + LAMBDA * sum_cable k * max(0, |x*-o| - r)               [prestress cost:
//                                                                the actual tension
//                                                                force carried by
//                                                                every cable at
//                                                                equilibrium -- any
//                                                                unnecessary tension
//                                                                is charged]
//     + PENALTY * (#free nodes missing an actively-taut cable on EITHER side)
//                                                               [instability: the
//                                                                target must be a
//                                                                STABLE minimum --
//                                                                a mast tip pulled
//                                                                only from one
//                                                                direction can be
//                                                                shoved off target
//                                                                for free]
//
// Baseline B: the checker's own "halve every gap" reference (every cable's rest
// length is set to half the target-shape distance of its own two endpoints,
// ignoring stiffness and ignoring which side needs how much force). Score:
//   ratio = min(1000, 100*B/max(1e-6,F)) / 1000.
// -----------------------------------------------------------------------------

static const double MAXLEN     = 100000.0;
static const double FORCE_EPS  = 0.03;
static const double LAMBDA     = 0.05;
static const double PENALTY    = 2.5;
static const int    BIS_ITERS  = 60;

int n, s, c;
vector<ll> target_;
vector<array<ll,3>> struts_;     // u v d (unused directly beyond grounding)
vector<array<ll,3>> cablesRaw;   // u v k as given in input order

vector<int> parent_;
int find_(int x){ while (parent_[x] != x) x = parent_[x] = parent_[parent_[x]]; return x; }

struct CableRef { int freeNode; double o; double k; int idx; }; // idx = position in output order
vector<vector<CableRef>> incident; // per free node

vector<char> grounded_;

void buildStructure() {
    parent_.resize(n);
    for (int i = 0; i < n; i++) parent_[i] = i;
    grounded_.assign(n, 0);
    grounded_[0] = 1;
    for (auto &e : struts_) {
        int u = (int)e[0], v = (int)e[1];
        parent_[find_(v)] = find_(u);
        grounded_[v] = 1; // by construction (validated), struts always attach a free/new node to a grounded one
    }
    incident.assign(n, {});
    for (int i = 0; i < c; i++) {
        int u = (int)cablesRaw[i][0], v = (int)cablesRaw[i][1];
        double k = (double)cablesRaw[i][2];
        bool gu = grounded_[u], gv = grounded_[v];
        if (gu == gv) continue; // degenerate (shouldn't happen); contributes nothing structural
        int freeNode = gu ? v : u;
        int gNode    = gu ? u : v;
        incident[freeNode].push_back({freeNode, (double)target_[gNode], k, i});
    }
}

// derivative of e(x) at x for one free node's incident list, given rest lengths r[]
double deriv(const vector<CableRef>& lst, const vector<double>& rest, double x) {
    double d = 0;
    for (auto &cr : lst) {
        double gap = x - cr.o;
        double ag = fabs(gap);
        double f = max(0.0, ag - rest[cr.idx]);
        if (gap > 0) d += cr.k * f;
        else if (gap < 0) d -= cr.k * f;
    }
    return d;
}

double solveFreeNode(const vector<CableRef>& lst, const vector<double>& rest) {
    double lo = -200000.0, hi = 200000.0;
    for (int it = 0; it < BIS_ITERS; it++) {
        double mid = 0.5 * (lo + hi);
        if (deriv(lst, rest, mid) < 0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Replays the whole structure for a given rest-length assignment; returns F.
double computeF(const vector<double>& rest) {
    double shapeErr = 0.0, tensionCost = 0.0;
    int unstable = 0;
    for (int v = 0; v < n; v++) {
        if (grounded_[v]) continue;
        const auto& lst = incident[v];
        if (lst.empty()) continue; // shouldn't happen
        double xs = solveFreeNode(lst, rest);
        shapeErr += (xs - (double)target_[v]) * (xs - (double)target_[v]);
        // A taut cable anchored at o < xs resists xs growing further RIGHT (its gap only
        // grows that way); a taut cable at o > xs resists xs growing further LEFT. Stability
        // (a true local minimum, not a marginal/floppy direction) needs BOTH guards present.
        bool hasGuardLeft = false, hasGuardRight = false;
        for (auto &cr : lst) {
            double gap = xs - cr.o;
            double f = max(0.0, fabs(gap) - rest[cr.idx]);
            double force = cr.k * f;
            tensionCost += force;
            if (force > FORCE_EPS) {
                if (gap < 0) hasGuardLeft = true;   // anchor to the right of xs
                else if (gap > 0) hasGuardRight = true; // anchor to the left of xs
            }
        }
        if (!(hasGuardLeft && hasGuardRight)) { unstable++; }
    }
    double F = shapeErr + LAMBDA * tensionCost + PENALTY * (double)unstable;
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    s = inf.readInt();
    c = inf.readInt();
    target_.resize(n);
    for (int i = 0; i < n; i++) target_[i] = inf.readLong();
    struts_.resize(s);
    for (int i = 0; i < s; i++) {
        ll u = inf.readLong(), v = inf.readLong(), d = inf.readLong();
        struts_[i] = {u, v, d};
    }
    cablesRaw.resize(c);
    for (int i = 0; i < c; i++) {
        ll u = inf.readLong(), v = inf.readLong(), k = inf.readLong();
        cablesRaw[i] = {u, v, k};
    }

    buildStructure();

    // ---- internal baseline B: "rest length = target-shape gap, exactly" ----
    // The literal naive idea -- make each cable exactly as long as the target
    // wants the gap to be. Every cable then carries EXACTLY ZERO tension at the
    // target, so no free node has a taut guard on either side: the target shape
    // is technically reached but is not a stable prestressed equilibrium at all
    // (this is the textbook tensegrity trap the family is built to punish).
    vector<double> restB(c);
    for (int i = 0; i < c; i++) {
        ll u = cablesRaw[i][0], v = cablesRaw[i][1];
        double gap = fabs((double)target_[v] - (double)target_[u]);
        restB[i] = gap;
    }
    double B = computeF(restB);
    if (B <= 0) B = 1.0;

    // ---- read participant output: c non-negative reals ----
    vector<double> rest(c);
    for (int i = 0; i < c; i++) {
        double r = ouf.readDouble(0.0, MAXLEN, "rest_length");
        if (!isfinite(r)) quitf(_wa, "non-finite rest length at index %d", i);
        rest[i] = r;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double F = computeF(rest);

    double sc = min(1000.0, 100.0 * B / max(1e-6, F));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}

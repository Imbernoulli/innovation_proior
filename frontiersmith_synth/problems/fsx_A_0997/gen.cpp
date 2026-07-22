#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Sixteen-Load Truss Margin"  (generator)  family: multi-load-truss-margin
//
// A simply-supported 2D pin-jointed bridge "ground structure": two chords
// (bottom y=0, top y=H) over W bays (span SPAN each), plus verticals and BOTH
// diagonal directions in every bay (a full candidate ground structure -- the
// participant selects a subset + a cross-section class per selected member,
// under a total-cost budget). K=16 load cases are swept: 1 gravity (every
// node), 4 wind cases on the top chord (uniform +x, an alternating
// checkerboard +x/-x by bay, and two differently-shaped ramps), and 11 moving
// concentrated point loads cycled round-robin over the bottom chord's
// interior nodes. Difficulty grows purely through W (more, narrower bays ->
// a harder combinatorial selection with the same per-load-case physics,
// see LOADMULT below) plus one needle case (test 7) that spikes a single
// moving-load position hard.
//
// PLANTED TRAP: gravity is symmetric & purely vertical. A rectangular bay with
// NO diagonal is a racking mechanism (1 kinematic DOF) whose displacement mode
// is purely horizontal -- gravity (all-vertical load) does zero work through
// that mode, so a structure optimized/verified against gravity ALONE can look
// perfectly fine (finite, low stress) while being a literal mechanism (score 0)
// under any of the wind cases, which DO have horizontal load components that
// excite exactly that racking mode. A solver that sizes every member from a
// single dominant load case (gravity) can therefore leave some bay completely
// unbraced and still look great on that one case -- and crater on 4+ others.
// The insight (shared-subspace): analyze ALL 16 cases (or at least several
// directionally-diverse ones) to find which members carry significant load
// across MANY cases simultaneously and prioritize exactly those -- generally
// the bracing nearest the two supports, and in some bays BOTH diagonal
// directions (X-bracing), because opposite wind directions route load through
// opposite diagonals in the same bay.
//
// Output:
//   W K
//   N M
//   SPAN H E SIGMA_Y
//   NAREA a_1 .. a_NAREA
//   BUDGET
//   PIN_NODE ROLLER_NODE
//   x_0 y_0 .. x_{N-1} y_{N-1}          (node coordinates, 0-indexed)
//   u_0 v_0 .. u_{M-1} v_{M-1}          (candidate members, 0-indexed node ids)
//   K
//   L f0 fx0 fy0 f1 fx1 fy1 ...         (per load case: L point loads)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const int WLADDER[11] = {0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14};
    int W = WLADDER[testId];

    // Total span and height are held fixed across the difficulty ladder: growing
    // testId adds MORE, NARROWER bays (a harder topology-selection / combinatorial
    // problem) without blowing up absolute force magnitudes (which would otherwise
    // scale like span^2 and make the checker's fixed-area baseline collapse at
    // large W). Total applied load is likewise held fixed and spread over more
    // nodes, so per-load-case physics stays comparable across the whole ladder.
    const double TOTAL_SPAN = 48.0, H = 6.0;
    const double SPAN = TOTAL_SPAN / W;
    const double E = 2000.0, SIGMA_Y = 250.0;
    const int NAREA = 3;
    double areaScale = 1.0 + 0.03 * (testId - 1);
    double area[3] = {1.0 * areaScale, 2.0 * areaScale, 4.0 * areaScale};

    int N = 2 * (W + 1);
    // node ids: bottom_i = i (0..W), top_i = (W+1)+i (0..W)
    int PIN = 0;      // bottom_0 : pin
    int ROLLER = W;   // bottom_W : roller

    vector<double> X(N), Y(N);
    for (int i = 0; i <= W; i++) { X[i] = i * SPAN; Y[i] = 0.0; }
    for (int i = 0; i <= W; i++) { X[(W + 1) + i] = i * SPAN; Y[(W + 1) + i] = H; }

    vector<pair<int,int>> mem;
    for (int i = 0; i < W; i++) mem.push_back({i, i + 1});                       // bottom chord
    for (int i = 0; i < W; i++) mem.push_back({(W + 1) + i, (W + 1) + i + 1});   // top chord
    for (int i = 0; i <= W; i++) mem.push_back({i, (W + 1) + i});                 // vertical
    for (int i = 0; i < W; i++) mem.push_back({i, (W + 1) + i + 1});              // diagonal A (bottom_i -> top_{i+1})
    for (int i = 0; i < W; i++) mem.push_back({(W + 1) + i, i + 1});              // diagonal B (top_i -> bottom_{i+1})
    int M = (int)mem.size();

    // ---- budget: 1.6x the cost of "chords + verticals + one diagonal/bay" at the
    //      smallest area class -- enough slack to X-brace SOME bays but not all. ----
    double diagLen = sqrt(SPAN * SPAN + H * H); // = 5.0
    double costMin = (2.0 * W * SPAN + (W + 1) * H + W * diagLen) * area[0];
    double budgetMul = 1.2;
    double BUDGET = budgetMul * costMin;

    // ---- 16 load cases ----
    const int K = 16;
    vector<vector<pair<int,double>>> caseFx(K), caseFy(K); // unused placeholder (kept simple below)
    vector<vector<int>> Lnode(K);
    vector<vector<double>> Lfx(K), Lfy(K);

    // Total gravity / wind load is fixed and spread over the (growing) node count,
    // so per-node magnitude shrinks as W grows -- absolute force levels stay
    // comparable across the difficulty ladder (see note above). LOADMULT is a
    // per-test calibration so the checker's fixed reference construction (chords +
    // verticals + one diagonal/bay, smallest area) lands at a thin-but-positive
    // stress margin on every test regardless of how member-force levels shift with
    // truss geometry and case mix.
    static const double LOADMULT[11] = {0, 3.21, 3.59, 3.51, 3.76, 3.71, 3.81, 2.75, 4.01, 4.11, 4.21};
    double loadMul = LOADMULT[testId];
    double G_TOT = 60.0 * loadMul;
    double WIND_TOT = 50.0 * loadMul;
    double gravity = G_TOT / N;
    double windBase = WIND_TOT / (W + 1);
    double moveMag = 34.0 * loadMul;

    // case 0: gravity, every node, fy = -gravity
    for (int i = 0; i < N; i++) { Lnode[0].push_back(i); Lfx[0].push_back(0.0); Lfy[0].push_back(-gravity); }

    // case 1: wind uniform, +x, on all top nodes
    for (int i = 0; i <= W; i++) { Lnode[1].push_back((W + 1) + i); Lfx[1].push_back(windBase); Lfy[1].push_back(0.0); }
    // case 2: wind CHECKERBOARD -- alternating +x/-x by bay index on the top chord.
    // (A uniform -x case would be an exact global sign-flip of case 1: for a LINEAR
    // elastic solve, F -> -F forces u -> -u and |stress| is therefore IDENTICAL to
    // case 1 for every possible design, wasting the case. The alternating pattern is
    // not a scalar multiple of case 1 for any structure, so it is genuinely distinct
    // -- and it stresses EACH bay's diagonal(s) in local shear independently, which
    // is a sharper test of per-bay bidirectional bracing than a single global push.)
    for (int i = 0; i <= W; i++) {
        double sgn = (i % 2 == 0) ? 1.0 : -1.0;
        Lnode[2].push_back((W + 1) + i); Lfx[2].push_back(sgn * windBase); Lfy[2].push_back(0.0);
    }
    // case 3: wind ramp, stronger at right end, +x
    for (int i = 0; i <= W; i++) {
        double frac = (W == 0) ? 1.0 : (double)i / W;
        double mag = windBase * (0.4 + 1.2 * frac);
        Lnode[3].push_back((W + 1) + i); Lfx[3].push_back(mag); Lfy[3].push_back(0.0);
    }
    // case 4: wind ramp, stronger at left end, +x
    for (int i = 0; i <= W; i++) {
        double frac = (W == 0) ? 1.0 : (double)i / W;
        double mag = windBase * (1.6 - 1.2 * frac);
        Lnode[4].push_back((W + 1) + i); Lfx[4].push_back(mag); Lfy[4].push_back(0.0);
    }

    // cases 5..15 (11 cases): moving concentrated load on bottom chord. Positions
    // cycle round-robin through the W-1 available interior nodes (1..W-1) so they
    // spread as evenly as possible; only truly tiny W (2 or 3 interior positions)
    // forces repeats, which is unavoidable geometry, not a generator bug.
    int needleTest = 7;
    int nInterior = max(1, W - 1);
    for (int j = 0; j < 11; j++) {
        int node = 1 + (j % nInterior);
        double mag = moveMag;
        // NEEDLE: on the designated needle test, spike ONE specific position hard
        if (testId == needleTest && j == 5) mag *= 1.6;
        Lnode[5 + j].push_back(node);
        Lfx[5 + j].push_back(0.0);
        Lfy[5 + j].push_back(-mag);
    }

    printf("%d %d\n", W, K);
    printf("%d %d\n", N, M);
    printf("%.6f %.6f %.6f %.6f\n", SPAN, H, E, SIGMA_Y);
    printf("%d", NAREA);
    for (int i = 0; i < NAREA; i++) printf(" %.6f", area[i]);
    printf("\n");
    printf("%.6f\n", BUDGET);
    printf("%d %d\n", PIN, ROLLER);
    for (int i = 0; i < N; i++) printf("%.6f %.6f\n", X[i], Y[i]);
    for (int i = 0; i < M; i++) printf("%d %d\n", mem[i].first, mem[i].second);
    printf("%d\n", K);
    for (int c = 0; c < K; c++) {
        int L = (int)Lnode[c].size();
        printf("%d", L);
        for (int j = 0; j < L; j++) printf(" %d %.6f %.6f", Lnode[c][j], Lfx[c][j], Lfy[c][j]);
        printf("\n");
    }
    return 0;
}

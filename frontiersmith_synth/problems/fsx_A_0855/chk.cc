#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Punching Holes Without Dropping the Sheet".
//
// Input:  R C N T ; then R grid rows ('#'=frame permanent, '.'=punch job,
//         'x'=void, never material) ; then N lines "r c tool" (0-indexed,
//         tool in [1,T]) listing the punch jobs in row-major scan order --
//         this listing order is exactly what output indices 1..N refer to.
// Output: a permutation p_1..p_N of 1..N -- the punch order.
//
// Objective (MIN): F = (turret rotation: circular distance between consecutive
//   punches' tools, T-position dial, starting parked at tool 1) + T * (number
//   of distinct cells that ever become newly stranded from the frame -- charged
//   once, the first time each loses connectivity; already-scrap cells are never
//   re-charged).
//
// Baseline B (checker-computed): the SAME objective for the trivial listing
//   order 1,2,...,N (no reasoning about tools or geometry). This is exactly
//   what the trivial reference reproduces (-> ratio 0.1).
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int R, C, N, T;
vector<string> grid;
vector<int> jr, jc, jt;   // 1-indexed job r,c,tool

// Simulate a given punch order (0-indexed job indices); returns F.
ll simulate(const vector<int>& order) {
    vector<vector<char>> present(R, vector<char>(C, 0));
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            present[r][c] = (grid[r][c] != 'x') ? 1 : 0;

    vector<vector<char>> penalized(R, vector<char>(C, 0));
    vector<int> qr(R * C), qc(R * C);
    vector<vector<char>> seen(R, vector<char>(C, 0));

    ll totalRot = 0, violationCost = 0;
    int prevTool = 1; // home / parked position
    const int dr[4] = {1, -1, 0, 0}, dc[4] = {0, 0, 1, -1};

    for (int idx : order) {
        int r = jr[idx], c = jc[idx], tool = jt[idx];
        int d = abs(tool - prevTool);
        d = min(d, T - d);
        totalRot += d;
        prevTool = tool;
        present[r][c] = 0;

        // BFS from anchor (0,0), which is always '#' (permanent, present).
        for (int rr = 0; rr < R; rr++) fill(seen[rr].begin(), seen[rr].end(), 0);
        int head = 0, tail = 0;
        qr[tail] = 0; qc[tail] = 0; tail++;
        seen[0][0] = 1;
        int reached = 0;
        while (head < tail) {
            int cr = qr[head], cc = qc[head]; head++;
            reached++;
            for (int k = 0; k < 4; k++) {
                int nr = cr + dr[k], nc = cc + dc[k];
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                if (!present[nr][nc] || seen[nr][nc]) continue;
                seen[nr][nc] = 1;
                qr[tail] = nr; qc[tail] = nc; tail++;
            }
        }
        int totalPresent = 0;
        for (int rr = 0; rr < R; rr++)
            for (int cc = 0; cc < C; cc++)
                if (present[rr][cc]) totalPresent++;
        if (reached < totalPresent) {
            for (int rr = 0; rr < R; rr++)
                for (int cc = 0; cc < C; cc++)
                    if (present[rr][cc] && !seen[rr][cc] && !penalized[rr][cc]) {
                        penalized[rr][cc] = 1;
                        violationCost += T;
                    }
        }
    }
    return totalRot + violationCost;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    R = inf.readInt();
    C = inf.readInt();
    N = inf.readInt();
    T = inf.readInt();
    grid.resize(R);
    for (int r = 0; r < R; r++) grid[r] = inf.readToken();

    jr.resize(N); jc.resize(N); jt.resize(N);
    for (int i = 0; i < N; i++) {
        jr[i] = inf.readInt(0, R - 1, "job_r");
        jc[i] = inf.readInt(0, C - 1, "job_c");
        jt[i] = inf.readInt(1, T, "job_tool");
    }

    // ---- internal baseline B: punch jobs in raw listing order ----
    vector<int> baseOrder(N);
    for (int i = 0; i < N; i++) baseOrder[i] = i;
    ll B = simulate(baseOrder);
    if (B <= 0) B = 1;

    // ---- read + validate the participant's permutation ----
    vector<char> used(N + 1, 0);
    vector<int> order(N);
    for (int k = 0; k < N; k++) {
        int p = ouf.readInt(1, N, "job_index");
        if (used[p]) quitf(_wa, "job %d punched more than once", p);
        used[p] = 1;
        order[k] = p - 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the permutation");

    ll F = simulate(order);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}

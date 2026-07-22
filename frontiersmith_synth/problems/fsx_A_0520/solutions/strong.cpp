// TIER: strong
// The insight: you are graded on the WORST season, so cluster against the worst-case
// envelope, not the pooled average. Two ideas the greedy recipe misses:
//   (1) initialise/weight on the max-over-seasons demand m_i = max_k w[i][k], so the
//       spread zones that set the max are never invisible;
//   (2) L1 cost wants coordinate-wise MEDIANS, and after each assignment we UPWEIGHT the
//       currently-worst season, so depots migrate to hedge the season that sets F.
// Multi-restart; keep the layout with the smallest actual worst-season cost.
#include <bits/stdc++.h>
using namespace std;

static int N, P, K;
static vector<long long> X, Y;
static vector<vector<long long>> W;   // W[i][k]
static vector<long long> M;           // envelope max_k

// actual objective for a layout: max_k sum_i w[i][k]*nearest_L1
static long long evalF(const vector<long long>& dx, const vector<long long>& dy, int wk = -1, vector<long long>* costOut = nullptr) {
    vector<long long> cost(K, 0);
    int PP = (int)dx.size();
    for (int i = 0; i < N; i++) {
        long long best = LLONG_MAX;
        for (int p = 0; p < PP; p++) {
            long long dd = llabs(X[i]-dx[p]) + llabs(Y[i]-dy[p]);
            if (dd < best) best = dd;
        }
        for (int k = 0; k < K; k++) cost[k] += W[i][k] * best;
    }
    if (costOut) *costOut = cost;
    long long f = 0; for (int k = 0; k < K; k++) f = max(f, cost[k]);
    return f;
}

int main() {
    if (scanf("%d %d %d", &N, &P, &K) != 3) return 0;
    X.assign(N,0); Y.assign(N,0); W.assign(N, vector<long long>(K,0)); M.assign(N,0);
    for (int i = 0; i < N; i++) {
        scanf("%lld %lld", &X[i], &Y[i]);
        long long mx = 0;
        for (int k = 0; k < K; k++) { scanf("%lld", &W[i][k]); mx = max(mx, W[i][k]); }
        M[i] = mx;
    }
    const long long C = 100000;

    vector<long long> bestDX, bestDY;
    long long bestF = LLONG_MAX;

    for (int restart = 0; restart < 8; restart++) {
        mt19937 rng(1000u + 7919u * (unsigned)restart);
        // ---- weighted k-means++ init on the envelope weight M ----
        vector<long long> dx(P), dy(P);
        {
            vector<double> dist(N, 1e30);
            double tot = 0; for (int i = 0; i < N; i++) tot += (double)M[i] + 1e-9;
            double r = uniform_real_distribution<double>(0, tot)(rng);
            int first = 0; double acc = 0;
            for (int i = 0; i < N; i++) { acc += (double)M[i] + 1e-9; if (acc >= r) { first = i; break; } }
            dx[0] = X[first]; dy[0] = Y[first];
            for (int c = 1; c < P; c++) {
                for (int i = 0; i < N; i++) {
                    double dd = (double)(llabs(X[i]-dx[c-1]) + llabs(Y[i]-dy[c-1]));
                    dd = dd*dd;
                    if (dd < dist[i]) dist[i] = dd;
                }
                double wtot = 0; for (int i = 0; i < N; i++) wtot += ((double)M[i] + 1e-9) * dist[i];
                if (wtot <= 0) { dx[c] = X[c % N]; dy[c] = Y[c % N]; continue; }
                double rr = uniform_real_distribution<double>(0, wtot)(rng);
                double a2 = 0; int pick = 0;
                for (int i = 0; i < N; i++) { a2 += ((double)M[i] + 1e-9) * dist[i]; if (a2 >= rr) { pick = i; break; } }
                dx[c] = X[pick]; dy[c] = Y[pick];
            }
        }

        vector<int> asg(N, 0);
        for (int it = 0; it < 30; it++) {
            // assignment (L1 nearest)
            for (int i = 0; i < N; i++) {
                long long best = LLONG_MAX; int bj = 0;
                for (int p = 0; p < P; p++) {
                    long long dd = llabs(X[i]-dx[p]) + llabs(Y[i]-dy[p]);
                    if (dd < best) { best = dd; bj = p; }
                }
                asg[i] = bj;
            }
            // find current worst season under this layout
            vector<long long> cost;
            long long f = evalF(dx, dy, -1, &cost);
            int worst = 0; for (int k = 0; k < K; k++) if (cost[k] > cost[worst]) worst = k;
            if (f < bestF) { bestF = f; bestDX = dx; bestDY = dy; }

            // reweight: envelope + strong pull toward the worst season, so depots hedge it
            // update each depot to the coordinate-wise WEIGHTED MEDIAN of its cluster (L1-optimal)
            double lambda = 4.0;
            for (int p = 0; p < P; p++) {
                vector<pair<long long,double>> vx, vy;
                double wsum = 0;
                for (int i = 0; i < N; i++) if (asg[i] == p) {
                    double ww = (double)M[i] + lambda * (double)W[i][worst] + 1e-6;
                    vx.push_back({X[i], ww}); vy.push_back({Y[i], ww}); wsum += ww;
                }
                if (vx.empty()) continue;
                auto wmed = [&](vector<pair<long long,double>>& v) -> long long {
                    sort(v.begin(), v.end());
                    double half = wsum / 2.0, run = 0;
                    for (auto& pr : v) { run += pr.second; if (run >= half) return pr.first; }
                    return v.back().first;
                };
                dx[p] = wmed(vx);
                dy[p] = wmed(vy);
            }
        }
        // final eval of the converged layout
        long long f = evalF(dx, dy);
        if (f < bestF) { bestF = f; bestDX = dx; bestDY = dy; }
    }

    if (bestDX.empty()) { bestDX.assign(P, C/2); bestDY.assign(P, C/2); }
    for (int p = 0; p < P; p++) {
        long long ox = min(C, max(0LL, bestDX[p]));
        long long oy = min(C, max(0LL, bestDY[p]));
        printf("%lld %lld\n", ox, oy);
    }
    return 0;
}

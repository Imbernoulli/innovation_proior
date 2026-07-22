// TIER: strong
// Insight: phase-only pattern synthesis is a projection problem, not a
// one-shot inversion. Alternate between
//   (1) the DESIRED complex far field: replace the realized magnitude at each
//       angle by the target/threshold-consistent magnitude but KEEP the
//       currently realized phase there (classic Gerchberg-Saxton move), and
//   (2) the FIXED-MAGNITUDE aperture constraint: solve the weighted
//       least-squares excitation that best reproduces that desired field
//       (an exact small NxN complex linear solve, not an approximate
//       unnormalized adjoint), then re-project every element's excitation
//       onto ITS OWN fixed-magnitude circle (keep a_i, take the phase).
// Null angles get an adaptively-boosted weight (grows with the current
// violation, like a reweighted-least-squares / soft dual step) so leaks that
// persist get pushed down harder each round. The best F seen is kept.
#include <bits/stdc++.h>
using namespace std;
typedef complex<double> cd;
static const double PI = 3.14159265358979323846;

static int N, M, K;
static double d, Pmax;
static vector<double> a;
static vector<double> thetaRad;
static vector<double> T, W;
static double sumW;
static vector<int> nullIdx;
static double thresh;
static vector<vector<cd>> eTab;   // eTab[i][m] = exp(j*2*pi*d*i*cos(theta_m))

static int quantize(double rad){
    double deg = rad * 180.0 / PI;
    long long p = (long long)llround(deg * 10.0);
    p %= 3600; if (p < 0) p += 3600;
    return (int)p;
}

static double computeF(const vector<int>& phi, vector<double>& P){
    for (int m = 0; m < M; m++){
        double re = 0, im = 0;
        for (int i = 0; i < N; i++){
            double ph = phi[i] * PI / 1800.0;
            cd x = a[i] * cd(cos(ph), sin(ph));
            cd c = x * eTab[i][m];
            re += c.real(); im += c.imag();
        }
        P[m] = (re * re + im * im) / (Pmax * Pmax);
    }
    double err = 0;
    for (int m = 0; m < M; m++) err += W[m] * (P[m] - T[m]) * (P[m] - T[m]);
    err /= sumW;
    double pen = 0;
    for (int idx : nullIdx){
        double over = P[idx] - thresh;
        if (over > 0) pen += over * over;
    }
    pen *= 4.0 / max(1, K);
    return err + pen;
}

// solve n x n complex linear system G x = b (Gaussian elimination, partial pivot)
static vector<cd> solveComplex(vector<vector<cd>> G, vector<cd> b, int n){
    for (int col = 0; col < n; col++){
        int piv = col; double best = abs(G[col][col]);
        for (int r = col + 1; r < n; r++){
            double v = abs(G[r][col]);
            if (v > best){ best = v; piv = r; }
        }
        if (piv != col){ swap(G[piv], G[col]); swap(b[piv], b[col]); }
        cd diag = G[col][col];
        if (abs(diag) < 1e-9) diag = cd(1e-9, 0);
        for (int r = col + 1; r < n; r++){
            cd factor = G[r][col] / diag;
            if (factor == cd(0, 0)) continue;
            for (int c = col; c < n; c++) G[r][c] -= factor * G[col][c];
            b[r] -= factor * b[col];
        }
    }
    vector<cd> x(n);
    for (int i = n - 1; i >= 0; i--){
        cd s = b[i];
        for (int j = i + 1; j < n; j++) s -= G[i][j] * x[j];
        cd diag = G[i][i]; if (abs(diag) < 1e-9) diag = cd(1e-9, 0);
        x[i] = s / diag;
    }
    return x;
}

int main(){
    int D1000;
    scanf("%d %d %d", &N, &M, &D1000);
    d = D1000 / 1000.0;
    a.resize(N);
    for (int i = 0; i < N; i++){ int x; scanf("%d", &x); a[i] = x / 1000.0; }
    Pmax = 0; for (double v : a) Pmax += v; if (Pmax <= 0) Pmax = 1;

    vector<int> ang(M);
    for (int m = 0; m < M; m++) scanf("%d", &ang[m]);
    thetaRad.resize(M);
    for (int m = 0; m < M; m++) thetaRad[m] = ang[m] * PI / 180000.0;

    T.resize(M);
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); T[m] = x / 10000.0; }
    W.resize(M); sumW = 0;
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); W[m] = x; sumW += x; }
    if (sumW <= 0) sumW = 1;

    scanf("%d", &K);
    nullIdx.resize(K);
    for (int i = 0; i < K; i++){ int x; scanf("%d", &x); nullIdx[i] = x - 1; }
    int th; scanf("%d", &th); thresh = th / 10000.0;

    eTab.assign(N, vector<cd>(M));
    for (int i = 0; i < N; i++)
        for (int m = 0; m < M; m++){
            double bp = 2.0 * PI * d * i * cos(thetaRad[m]);
            eTab[i][m] = cd(cos(bp), sin(bp));
        }

    vector<double> desiredP(M);
    for (int m = 0; m < M; m++) desiredP[m] = T[m];
    vector<char> isNull(M, 0);
    for (int idx : nullIdx){ isNull[idx] = 1; desiredP[idx] = min(T[idx], thresh); }

    vector<int> phi(N, 0), best = phi;
    vector<double> P(M);
    double bestF = computeF(phi, P);

    double regEps = 1e-6 * max(1.0, sumW);
    const int ITERS = 22;
    for (int iter = 0; iter < ITERS; iter++){
        computeF(phi, P);   // refresh P for current phi

        vector<double> useW(M);
        for (int m = 0; m < M; m++){
            useW[m] = W[m];
            if (isNull[m]){
                double over = P[m] - thresh;
                if (over > 0) useW[m] += 4000.0 * over;
            }
        }

        // desired complex field: target/threshold magnitude, KEEP current phase
        vector<cd> Dm(M);
        for (int m = 0; m < M; m++){
            double re = 0, im = 0;
            for (int i = 0; i < N; i++){
                double ph = phi[i] * PI / 1800.0;
                cd x = a[i] * cd(cos(ph), sin(ph));
                cd c = x * eTab[i][m];
                re += c.real(); im += c.imag();
            }
            double phase = (re == 0 && im == 0) ? 0.0 : atan2(im, re);
            double mag = sqrt(max(0.0, desiredP[m])) * Pmax;
            Dm[m] = polar(mag, phase);
        }

        // exact weighted least-squares excitation for that desired field
        vector<vector<cd>> G(N, vector<cd>(N, cd(0, 0)));
        vector<cd> b(N, cd(0, 0));
        for (int m = 0; m < M; m++){
            double w = useW[m];
            if (w <= 0) continue;
            for (int i = 0; i < N; i++){
                cd ce = conj(eTab[i][m]);
                b[i] += w * Dm[m] * ce;
                for (int j = 0; j < N; j++) G[i][j] += w * eTab[j][m] * ce;
            }
        }
        for (int i = 0; i < N; i++) G[i][i] += regEps;

        vector<cd> x = solveComplex(G, b, N);

        // project onto the fixed-magnitude aperture constraint: keep phase only
        vector<int> nphi(N);
        for (int i = 0; i < N; i++){
            if (abs(x[i]) < 1e-12) nphi[i] = phi[i];
            else nphi[i] = quantize(arg(x[i]));
        }
        phi = nphi;

        double f = computeF(phi, P);
        if (f < bestF){ bestF = f; best = phi; }
    }

    for (int i = 0; i < N; i++) printf("%d%c", best[i], i + 1 == N ? '\n' : ' ');
    return 0;
}

# Winograd minimal-filtering fast convolution for convnets

## Problem

A convnet layer correlates K filters (C channels, R×S) against N images (C channels, H×W). Direct convolution spends R·S multiplications per output (9 for a 3×3 filter), so the layer costs about R·S·N·H'·W'·C·K multiplications — the bottleneck for both training and low-latency inference. FFT convolution helps only for large filters, large tiles, and large batches; the dominant 3×3 small-batch regime needs something else. The goal: compute the *exact* convolution with far fewer multiplications, keeping the heavy stage a dense matrix multiply that is efficient even at batch size 1, with a small memory workspace.

## Key idea

Linear convolution is polynomial multiplication. Computing m outputs of an r-tap FIR filter, F(m,r), needs a minimum of µ(F(m,r)) = m+r−1 general multiplications (one per distinct input the m-window touches), versus m·r for direct. The minimum is reached by Cook–Toom: evaluate filter and data polynomials at m+r−1 points (small integers 0, ±1, ±2, … plus the point at infinity), multiply the values pointwise (the only general multiplies), and Lagrange-interpolate the product back. Filtering is the transpose of linear convolution, so by Winograd's matrix-exchange the filtering algorithm is

  Y = Aᵀ [ (G g) ⊙ (Bᵀ d) ]   (1D),

with data-independent transforms: filter transform G (r→α), data transform Bᵀ (α→α), elementwise multiply (⊙), inverse transform Aᵀ (α→m), α = m+r−1. Nesting a minimal 1D F(m,r) with F(n,s) gives a minimal 2D F(m×n, r×s) using (m+r−1)(n+s−1) multiplications:

  Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A   (2D).

For F(2×2, 3×3): 4×4 = 16 multiplies versus direct 36 → 2.25×, on a 4×4 tile. For F(4×4, 3×3): 6×6 = 36 versus 144 → 4×, on a 6×6 tile. The cost is paid in cheap additions and small constant-scalings, which grow quadratically with tile size, and the transform coefficients grow in magnitude with tile size (the F(4,3) filter transform already has 1/24 entries), degrading accuracy — so F(2×2,3×3) is the small, best-conditioned operating point, while F(4×4,3×3) is the larger speed/accuracy tradeoff when the precision budget allows it.

The layer-level payoff: sum over channels *inside* the transform domain (amortizing the inverse transform over C), so for each transform-domain coordinate (ξ,ν) ∈ {0..α−1}²,

  M^(ξ,ν) = Σ_c U^(ξ,ν)_{k,c} V^(ξ,ν)_{c,b} = U^(ξ,ν) V^(ξ,ν),

a (K×C)·(C×P) matrix multiply. The whole layer's heavy stage is α² independent dense GEMMs whose reduction dimension is channels, not batch — so it is efficient even at batch 1, and its multiply stage is exactly 1 real multiply per input, strictly beating FFT's ≥1.5 (with Hermitian symmetry + fast complex multiply).

## Final algorithm (forward layer, F(m×m, r×r))

1. Filter transform: U = G g Gᵀ for every filter/channel; scatter to α² matrices U^(ξ,ν) of shape K×C.
2. Data transform: divide each channel into α×α tiles overlapping by r−1; V = Bᵀ d B; scatter to α² matrices V^(ξ,ν) of shape C×P.
3. Multiply stage: α² batched GEMMs M^(ξ,ν) = U^(ξ,ν) V^(ξ,ν) (K×P).
4. Inverse transform: gather M back to α×α per output tile; Y_tile = Aᵀ M A (m×m).

Backprop reuses the same machinery: the input gradient is a convolution with flipped filters (same algorithm); the weight gradient F(R×S, H×W) is decomposed into a sum of small convolutions, e.g. F(3×3, 2×2) overlap-add.

## Code

The transforms above are the Cook–Toom construction at points {0,1,−1,∞}: evaluating filter and data polynomials at those points (the last "point at infinity" reading off the top coefficient), the interpolation denominators ∏_{k≠i}(β_i−β_k) producing the 1/2-scale entries of G, and transposing the linear-convolution factorization to get the filtering form Aᵀ[(Gg)⊙(Bᵀd)]. Expanding that form recovers exactly y₀=d₀g₀+d₁g₁+d₂g₂ and y₁=d₁g₀+d₂g₁+d₃g₂, four multiplies versus the direct six; the larger F(4,3) and F(3,2) transforms come from the same construction at {0,±1,±2,∞} and {0,±1,∞}, the F(4,3) filter transform acquiring the 1/24-scale entries.

The deliverable is a single self-contained C++17 program for the F(2×2,3×3) layer. It reads a convolution layer from stdin — `N C H W K`, then the N·C·H·W data values (index order [n][c][h][w]), then the K·C·3·3 filter values ([k][c][u][v]) — and writes the valid output Y[n][k][H−2][W−2] to stdout, one row of W−2 space-separated values per line. The heavy stage is the α²=16 dense matrix multiplies that fell out of the channel sum; each output costs 4 real multiplies versus the direct 9, and it matches a direct convolution to fp32 rounding.

```cpp
// Winograd minimal-filtering fast convolution F(2x2,3x3) for a convnet layer.
// Reads from stdin:  N C H W K, then the N*C*H*W data values (row-major,
//   indices [n][c][h][w]), then the K*C*3*3 filter values ([k][c][u][v]).
// Writes to stdout:  the valid layer output Y[n][k][H-2][W-2], one row of
//   (W-2) space-separated values per line, blocks ordered by n then k.
// The heavy stage is alpha^2 = 16 dense matrix multiplies whose reduction
// dimension is channels (efficient even at batch 1); each output costs 4
// real multiplies vs the direct 9.
#include <bits/stdc++.h>
using namespace std;

// F(2x2,3x3) transforms (the F(2,3) triple applied along both axes).
// Data transform B^T (4x4), filter transform G (4x3), inverse transform A^T (2x4).
static const double BT[4][4] = {
    {1, 0, -1, 0},
    {0, 1,  1, 0},
    {0, -1, 1, 0},
    {0, -1, 0, 1}};
static const double G[4][3] = {
    {1,    0,    0},
    {0.5,  0.5,  0.5},
    {0.5, -0.5,  0.5},
    {0,    0,    1}};
static const double AT[2][4] = {
    {1, 1,  1, 0},
    {0, 1, -1, 1}};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, C, H, W, K;
    if (!(cin >> N >> C >> H >> W >> K)) return 0;

    const int R = 3, S = 3, m = 2, r = 3, alpha = 4;
    // Data: D[n][c][h][w]
    vector<double> D((size_t)N * C * H * W);
    for (double &x : D) cin >> x;
    // Filters: Wt[k][c][u][v]
    vector<double> Wt((size_t)K * C * R * S);
    for (double &x : Wt) cin >> x;

    auto Didx = [&](int n, int c, int h, int w) -> double & {
        return D[(((size_t)n * C + c) * H + h) * W + w];
    };
    auto Widx = [&](int k, int c, int u, int v) -> double & {
        return Wt[(((size_t)k * C + c) * R + u) * S + v];
    };

    const int Ho = H - R + 1, Wo = W - S + 1;       // valid output size
    if (Ho <= 0 || Wo <= 0) return 0;
    const int tH = (Ho + m - 1) / m, tW = (Wo + m - 1) / m;  // tiles per dim
    const int P = N * tH * tW;                       // total tiles
    const int Hp = tH * m + r - 1, Wp = tW * m + r - 1;       // padded extent

    // Padded data Dp[n][c][hp][wp] (zero-filled beyond H,W).
    vector<double> Dp((size_t)N * C * Hp * Wp, 0.0);
    auto Dpidx = [&](int n, int c, int h, int w) -> double & {
        return Dp[(((size_t)n * C + c) * Hp + h) * Wp + w];
    };
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    Dpidx(n, c, h, w) = Didx(n, c, h, w);

    // Filter transform U = G g G^T, scattered into alpha*alpha matrices of
    // shape K x C.  U[(xi*alpha+nu)][k*C + c].
    vector<double> U((size_t)alpha * alpha * K * C, 0.0);
    {
        double tmp[4][3], u[4][4];
        for (int k = 0; k < K; k++)
            for (int c = 0; c < C; c++) {
                // tmp = G * g   (4x3)
                for (int i = 0; i < alpha; i++)
                    for (int j = 0; j < S; j++) {
                        double s = 0;
                        for (int t = 0; t < R; t++) s += G[i][t] * Widx(k, c, t, j);
                        tmp[i][j] = s;
                    }
                // u = tmp * G^T (4x4)
                for (int i = 0; i < alpha; i++)
                    for (int j = 0; j < alpha; j++) {
                        double s = 0;
                        for (int t = 0; t < S; t++) s += tmp[i][t] * G[j][t];
                        u[i][j] = s;
                    }
                for (int xi = 0; xi < alpha; xi++)
                    for (int nu = 0; nu < alpha; nu++)
                        U[((size_t)(xi * alpha + nu) * K + k) * C + c] = u[xi][nu];
            }
    }

    // Data transform V = B^T d B, scattered into alpha*alpha matrices of
    // shape C x P.  V[(xi*alpha+nu)][c*P + b].
    vector<double> V((size_t)alpha * alpha * C * P, 0.0);
    {
        double tmp[4][4], v[4][4];
        int b = 0;
        for (int n = 0; n < N; n++)
            for (int ti = 0; ti < tH; ti++)
                for (int tj = 0; tj < tW; tj++) {
                    for (int c = 0; c < C; c++) {
                        // tmp = B^T * tile  (4x4)
                        for (int i = 0; i < alpha; i++)
                            for (int j = 0; j < alpha; j++) {
                                double s = 0;
                                for (int t = 0; t < alpha; t++)
                                    s += BT[i][t] * Dpidx(n, c, ti * m + t, tj * m + j);
                                tmp[i][j] = s;
                            }
                        // v = tmp * B  (= tmp * (B^T)^T) (4x4)
                        for (int i = 0; i < alpha; i++)
                            for (int j = 0; j < alpha; j++) {
                                double s = 0;
                                for (int t = 0; t < alpha; t++) s += tmp[i][t] * BT[j][t];
                                v[i][j] = s;
                            }
                        for (int xi = 0; xi < alpha; xi++)
                            for (int nu = 0; nu < alpha; nu++)
                                V[((size_t)(xi * alpha + nu) * C + c) * P + b] = v[xi][nu];
                    }
                    b++;
                }
    }

    // Heavy stage: alpha^2 dense GEMMs  M^(xi,nu) = U^(xi,nu) (KxC) * V^(xi,nu) (CxP).
    // M[(xi*alpha+nu)][k*P + b].
    vector<double> M((size_t)alpha * alpha * K * P, 0.0);
    for (int e = 0; e < alpha * alpha; e++) {
        const double *Ue = &U[(size_t)e * K * C];
        const double *Ve = &V[(size_t)e * C * P];
        double *Me = &M[(size_t)e * K * P];
        for (int k = 0; k < K; k++)
            for (int c = 0; c < C; c++) {
                double a = Ue[(size_t)k * C + c];
                if (a == 0.0) continue;
                const double *Vrow = &Ve[(size_t)c * P];
                double *Mrow = &Me[(size_t)k * P];
                for (int b = 0; b < P; b++) Mrow[b] += a * Vrow[b];
            }
    }

    // Inverse transform Y_tile = A^T M A  (m x m) per (k, tile), scatter to output.
    vector<double> Y((size_t)N * K * tH * m * tW * m, 0.0);
    const int Yh = tH * m, Yw = tW * m;
    auto Yidx = [&](int n, int k, int h, int w) -> double & {
        return Y[(((size_t)n * K + k) * Yh + h) * Yw + w];
    };
    {
        double mt[4][4], tmp[2][4], y[2][2];
        for (int k = 0; k < K; k++) {
            int b = 0;
            for (int n = 0; n < N; n++)
                for (int ti = 0; ti < tH; ti++)
                    for (int tj = 0; tj < tW; tj++) {
                        for (int xi = 0; xi < alpha; xi++)
                            for (int nu = 0; nu < alpha; nu++)
                                mt[xi][nu] = M[((size_t)(xi * alpha + nu) * K + k) * P + b];
                        // tmp = A^T * M  (2x4)
                        for (int i = 0; i < m; i++)
                            for (int j = 0; j < alpha; j++) {
                                double s = 0;
                                for (int t = 0; t < alpha; t++) s += AT[i][t] * mt[t][j];
                                tmp[i][j] = s;
                            }
                        // y = tmp * A  (= tmp * (A^T)^T) (2x2)
                        for (int i = 0; i < m; i++)
                            for (int j = 0; j < m; j++) {
                                double s = 0;
                                for (int t = 0; t < alpha; t++) s += tmp[i][t] * AT[j][t];
                                y[i][j] = s;
                            }
                        for (int i = 0; i < m; i++)
                            for (int j = 0; j < m; j++)
                                Yidx(n, k, ti * m + i, tj * m + j) = y[i][j];
                        b++;
                    }
        }
    }

    // Emit the valid Ho x Wo region per (n, k).
    cout << setprecision(10);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            for (int i = 0; i < Ho; i++) {
                for (int j = 0; j < Wo; j++) {
                    if (j) cout << ' ';
                    cout << Yidx(n, k, i, j);
                }
                cout << '\n';
            }
    return 0;
}
```

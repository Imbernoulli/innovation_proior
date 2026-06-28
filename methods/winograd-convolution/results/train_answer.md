Deep recognition networks are now built almost entirely from small $3\times 3$ convolution layers, and those layers are where essentially all the compute goes — paid once across the GPU-days of training and again at every inference. A layer correlates $K$ filters of $C$ channels and size $R\times S$ against $N$ images of $C$ channels and size $H\times W$, and the direct way of doing it spends $R\cdot S$ multiplications per output pixel per channel — nine multiplies for a $3\times 3$ filter — so the whole layer costs on the order of $R\cdot S\cdot N\cdot H'\cdot W'\cdot C\cdot K$ multiplications. That multiply count is the wall. The two regimes that matter most are the worst for the existing fast methods: very small batches (a single image for low-latency detection, small per-node minibatches when training is sharded) and small filters ($3\times 3$ almost exclusively). The natural way to make a layer fast on a GPU is to cast it as a dense matrix multiply (im2col + GEMM), which is exact and needs no workspace but is minimal for nothing — it pays all nine multiplies. The FFT route turns convolution into pointwise multiplication in the Fourier domain and amortizes the transforms across many feature maps, but it is built for the opposite corner of the problem: its pointwise stage multiplies *complex* numbers (four real multiplies, three with a fast-complex-multiply trick, roughly halved again by Hermitian symmetry, but never below one real multiply per input), it needs large tiles to amortize the transform (a $64\times 64$ tile turns a nine-element filter into 4096 stored units and a multi-gigabyte workspace), and it needs a large batch to generate enough tiles to fill an efficient pointwise stage — exactly why a vendor library that flips to FFT at moderate batch can collapse below 2 TFLOPS. Strassen recursion on the layer cuts the *number* of convolutions by $8/7$ per step but halves all three matrix dimensions, shrinking precisely what a GEMM needs to be large; and low-precision quantization changes the datatype, not the multiply count. None of these reduces the arithmetic of a *small* filter while keeping the heavy stage a GEMM that stays efficient at batch one.

The instinct that nine multiplies per window is forced rests on computing a length-9 dot product the sliding way, forming each product and summing. But the outputs are bilinear forms in the data and the filter, and a bilinear form need not be realized by literally forming each monomial: one can take linear combinations of the inputs, multiply *those*, and arrange for the unwanted cross-terms to cancel in the recombination — exactly as $(a+ib)(c+id)$ needs only three real multiplies instead of four. On multiply-bound hardware where an add is nearly free, trading multiplies for adds is a pure win. I propose the Winograd minimal-filtering convolution, which realizes this trade at the proven arithmetic minimum. The load-bearing fact is that linear convolution *is* polynomial multiplication: writing the filter as $g(x)=g_0+g_1 x+g_2 x^2$ and the data as $d(x)$, the coefficients of $g(x)\,d(x)$ are the linear convolution of the two sequences. Computing $m$ outputs of an $r$-tap filter, denoted $F(m,r)$, therefore needs a minimum of $\mu(F(m,r)) = m+r-1$ general multiplications — one per distinct input value the $m$-output window touches — against the $m\cdot r$ of direct computation. The minimum is reached by Cook–Toom: a degree-$(m+r-2)$ result is pinned down by its values at $m+r-1$ points, so evaluate $g$ and $d$ at $m+r-1$ chosen points (small integers $0,\pm 1,\pm 2,\dots$ plus the "point at infinity," which simply reads off the top coefficient), multiply the values pointwise — these are the only general multiplies — and Lagrange-interpolate the product back. Choosing small integer points makes the evaluations and the interpolation cheap adds and scalings by small constants that depend only on the points, never on the data, so the matrices are built once. Asking for the full linear convolution would re-derive the direct count, because filtering wants only the $m$ valid outputs, not all $m+r-1$ product coefficients; the modified Cook–Toom move of taking the point at infinity reduces the degree to exactly what is needed, landing on $m+r-1$.

The piece that makes it a *filtering* algorithm rather than a *linear-convolution* one is the transpose. Filtering — a fixed filter slid to produce $m$ valid outputs — is the transpose of the linear-convolution map. Factor linear convolution as $T = C\,H\,D$ with $D$ a pre-addition matrix forming the input combinations, $H$ diagonal (its nonzero count is the multiply count, $m+r-1$ by construction), and $C$ the interpolation recombination; by Winograd's matrix-exchange theorem the filtering problem is computed by transposing this factorization, so the data and recombination roles swap and transpose. The end form in 1D is
$$Y = A^{\mathsf T}\,\big[\,(G\,g)\odot(B^{\mathsf T} d)\,\big],$$
where $\odot$ is elementwise multiply: the filter transform $G$ takes $g$ into the point-value domain, the data transform $B^{\mathsf T}$ takes $d$ into the same domain, the $\alpha=m+r-1$ elementwise products are the only real multiplies, and $A^{\mathsf T}$ interpolates them back to the $m$ outputs. For $F(2,3)$ with points $\{0,1,-1,\infty\}$ the transforms are $B^{\mathsf T}=\big[\begin{smallmatrix}1&0&-1&0\\0&1&1&0\\0&-1&1&0\\0&-1&0&1\end{smallmatrix}\big]$, $G=\big[\begin{smallmatrix}1&0&0\\1/2&1/2&1/2\\1/2&-1/2&1/2\\0&0&1\end{smallmatrix}\big]$, $A^{\mathsf T}=\big[\begin{smallmatrix}1&1&1&0\\0&1&-1&1\end{smallmatrix}\big]$, giving four multiplies versus the direct six, and expanding $A^{\mathsf T}[(Gg)\odot(B^{\mathsf T}d)]$ recovers exactly $y_0=d_0g_0+d_1g_1+d_2g_2$ and $y_1=d_1g_0+d_2g_1+d_3g_2$, with the price paid in additions. Because $G\,g$ depends only on the filter, and a convnet filter is reused across every tile of the image and the whole batch, the filter transform is computed once and amortized — only the data-side transform scales with the image.

Minimal 1D algorithms nest into minimal 2D ones: applying $F(m,r)$ along rows and $F(n,s)$ along columns gives $F(m\times n, r\times s)$ with $(m+r-1)(n+s-1)$ multiplies. For square $3\times 3$ filters,
$$Y = A^{\mathsf T}\,\big[\,(G\,g\,G^{\mathsf T})\odot(B^{\mathsf T} d\,B)\,\big]\,A,$$
where $g$ is the $r\times r$ filter, $d$ an $\alpha\times\alpha$ tile, $\alpha=m+r-1$: the filter transform $G\,g\,G^{\mathsf T}$ and the data transform $B^{\mathsf T} d\,B$ each produce $\alpha\times\alpha$ arrays, the elementwise product is $\alpha^2$ real multiplies, and $A^{\mathsf T}[\cdot]A$ interpolates back to an $m\times m$ output tile. $F(2\times 2,3\times 3)$ uses $4\times 4=16$ multiplies against the direct $36$ — a $2.25\times$ reduction — on a tiny $4\times 4$ tile, the opposite of the FFT's $64\times 64$. The layer-level payoff comes from the channel sum. Computing $A^{\mathsf T}[\cdot]A$ per (filter, channel, tile) and then summing over channels would pay the inverse transform $C$ times per output tile; but the channel sum and the inverse transform are both linear, so the sum can be pushed *inside* the transform domain and the inverse transform applied just once per output tile,
$$Y_{i,k,\mathrm{tile}} = A^{\mathsf T}\,\Big[\,\textstyle\sum_c (G\,g_{k,c}\,G^{\mathsf T})\odot(B^{\mathsf T} d_{c,i,\mathrm{tile}}\,B)\,\Big]\,A.$$
Writing $U_{k,c}=G\,g_{k,c}\,G^{\mathsf T}$ and $V_{c,b}=B^{\mathsf T} d_{c,b}\,B$ and fixing a single transform-domain coordinate $(\xi,\nu)\in\{0,\dots,\alpha-1\}^2$, the inner sum becomes
$$M^{(\xi,\nu)}_{k,b}=\sum_c U^{(\xi,\nu)}_{k,c}\,V^{(\xi,\nu)}_{c,b}=U^{(\xi,\nu)}\,V^{(\xi,\nu)},$$
a $(K\times C)\cdot(C\times P)$ dense matrix multiply. So the entire heavy stage of the layer is $\alpha^2$ independent GEMMs — 16 of them for $F(2\times 2,3\times 3)$ — whose reduction dimension is *channels*, not batch. That is the property I was chasing: the GEMMs stay efficient even at batch one, because a single image already supplies full-width matrices over $C$, where the FFT's pointwise stage starves without a large batch. And the multiply stage is exactly one real multiply per input, which strictly beats the FFT's complex-multiply stage — with Hermitian symmetry and the fast three-real-multiply complex product, $3(\lfloor\alpha/2\rfloor+1)/\alpha > 1.5$ real multiplies per input for every tile size — on multiplies, on workspace memory, and on the small-batch regime simultaneously.

The tile size cannot be cranked up freely to drive the normalized cost $\alpha'=(m+r-1)^2/m^2$ toward one. Two costs grow with the tile: the number of additions and constant-multiplications in the transforms grows quadratically, and — the sharper limit — the magnitudes of the transform-matrix entries grow, because adding interpolation points $2,-2,3,-3,\dots$ makes the Lagrange denominators $\prod_{k\neq i}(\beta_i-\beta_k)$ blow up. Carrying the CRT construction out to $F(4,3)$ with points $\{0,1,-1,2,-2,\infty\}$ already produces denominators $4, 6, 24$ — the filter transform $G$ acquires $1/24$-scale entries — and $F(6,3)$ throws up $1/720$; in finite precision such matrices lose bits. So there is a sweet spot at *small* tiles, not a limit at one: $F(2\times 2,3\times 3)$ (tile 4, transform entries only $0,\pm 1,\pm\tfrac12$, best conditioned, $2.25\times$) is the safe default, and $F(4\times 4,3\times 3)$ (tile 6, $36$ vs $144$, the full $4\times$) is available when the precision budget — convnets being known to tolerate surprisingly few bits — can absorb the larger roundoff. The same machinery covers backprop: the input gradient is a convolution with spatially flipped filters, so the forward algorithm applies unchanged; the weight gradient is $F(R\times S, H\times W)$, whose huge $H\times W$ "filter" is exactly the large tile to avoid, so it is decomposed into a sum of small convolutions — e.g. $F(3\times 3, 2\times 2)$, the tile-4 sibling of $F(2\times 2,3\times 3)$, overlap-added over the large operands. I deliberately do not stack Strassen on top: its $8/7$ gain comes at the cost of halving $K$, $C$, and $P$, the dimensions my $\alpha^2$ GEMMs need to be large, so it is a net loss here except possibly for layers with very large $C$, $K$, $P$.

The $G$, $B^{\mathsf T}$, $A^{\mathsf T}$ above are the Cook–Toom construction at points $\{0,1,-1,\infty\}$ — evaluate the filter and data polynomials at those points (the "point at infinity" reading off the top coefficient), divide by the interpolation denominators $\prod_{k\neq i}(\beta_i-\beta_k)$ (which place the $1/2$-scale entries into $G$), and transpose the linear-convolution factorization to land on the filtering form $A^{\mathsf T}[(Gg)\odot(B^{\mathsf T}d)]$; expanding it reproduces exactly $y_0=d_0g_0+d_1g_1+d_2g_2$ and $y_1=d_1g_0+d_2g_1+d_3g_2$. The same construction at $\{0,\pm1,\pm2,\infty\}$ gives the $F(4,3)$ transforms with the $1/24$-scale $G$ entries, and at $\{0,\pm1,\infty\}$ the $F(3,2)$ transforms for tiled weight-gradient accumulation.

The deliverable is a single self-contained C++17 program implementing the $F(2\times 2,3\times 3)$ layer — tile, transform, the $\alpha^2$ GEMMs that came out of the channel sum, and the single inverse transform. It reads a convolution layer from stdin (`N C H W K`, then the $N\cdot C\cdot H\cdot W$ data values in index order $[n][c][h][w]$, then the $K\cdot C\cdot 3\cdot 3$ filter values $[k][c][u][v]$) and writes the valid output $Y[n][k][H-2][W-2]$ to stdout, one row of $W-2$ space-separated values per line. Each output costs four real multiplies versus the direct nine, and it matches a direct convolution to fp32 rounding.

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

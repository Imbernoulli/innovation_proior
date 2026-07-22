Let me start from what actually goes wrong when I forecast a network of sensors. I have `N` series — traffic-speed loops on a highway, say — each `T` timestamps long, stacked into `X ∈ R^{N×T}`. From the last `K` columns I want the next `H`. The thing that makes this hard, and the thing every clean univariate forecaster ignores, is that the series are coupled: congestion on one segment shows up a few minutes later on the next, so series `i`'s future genuinely depends on series `j`'s recent past. I have two kinds of structure to capture — the temporal pattern *within* each series and the correlation *across* series — and the accuracy lives in the coupling between them, not in either one alone. So what do I have on the shelf, and exactly where does each tool stop short?

For the within-series part, the defaults are recurrent nets and temporal convolutions, and they're fine at it. There's also a frequency-domain lens I keep coming back to: take the discrete Fourier transform of a series, `y_u = (1/N) Σ_t x_t e^{-i2πut/N}`, and the periodic and autocorrelation structure that's smeared across the time axis collapses onto a handful of trigonometric coefficients. Periodicity in traffic is real — the daily rush-hour cycle — so a basis that names "frequency" directly is the right basis to model on. People have used DFT inside LSTMs for exactly this. For the across-series part, the natural object is a graph: nodes are series, edge weights `w_{ij}` are correlation strengths in an adjacency `W`. And on a graph there's a Fourier theory too. For a symmetric nonnegative `W`, the normalized Laplacian `L = I_N − D^{-1/2} W D^{-1/2}` is real, symmetric, PSD, so it diagonalizes, `L = U Λ U^T`, the eigenvectors `U` are "graph Fourier modes" over the nodes, and the graph Fourier transform of a node signal is `GF(x) = U^T x`, inverse `U x̂`. A spectral graph convolution filters in that basis, `y = U g_θ(Λ) U^T x`.

So I have a Fourier transform for the *temporal* axis (DFT) and a Fourier transform for the *spatial/cross-series* axis (GFT), and the state of the art — the stacked graph-conv-plus-recurrent models like DCRNN and STGCN — does the obvious thing: it interleaves a graph-convolution module with a temporal module. A graph step to mix across nodes, a GRU or gated temporal conv to march in time, alternate. And these are the models to beat on traffic. But stare at what they actually do. The graph module is spectral (it lives in the GFT basis), but the temporal module runs entirely in the *time* domain. The two computations never share a representation. I have two Fourier views available, and the architecture uses one of them spatially and throws the other away temporally. That feels like leaving the whole point of having a transform on the table. And there's a second sore spot: DCRNN and STGCN want the adjacency `W` handed to them as a prior — the road network. Sometimes I don't have a graph. And even when I do, the physical road network may not be the structure that actually drives the forecast; two roads can be far apart yet behave identically at rush hour. So I have two complaints to fix at once: model both axes in a *single* spectral representation, and learn the graph from the data instead of being told it.

Take the graph-learning problem first, because everything downstream needs a `W`. I don't want to hand-build it, so I want the model to produce `W ∈ R^{N×N}` from the data. What's a principled way to score how related two series are? Each series has a temporal signature; if I can compress node `i`'s history window into a vector `r_i`, then the affinity between `i` and `j` is some learned function of `r_i` and `r_j`, and a learned, asymmetric-then-symmetrized score over all pairs is exactly what self-attention gives me. I can describe the clean version as query-key attention, `Q = R W^Q`, `K = R W^K`, `W = softmax(QK^T / √d)`. A code-friendly variant makes the same idea additive: each node representation gets a scalar key and a scalar query, the pair score is `LeakyReLU(k_i + q_j)`, and a row-wise softmax turns the scores into normalized affinities. That `N×N` matrix *is* my adjacency. It's data-driven, it needs no prior topology, and it is interpretable: I can read off which sensors the model thinks drive which, and check it against the map.

Now the harder, central question: how do I model both axes jointly *in the spectral domain*, instead of stapling a spectral spatial module to a time-domain temporal one? Apply GFT first, to the whole multivariate input. `GF(X) = U^T X` projects the `N` coupled series onto the Laplacian eigenbasis. Projecting correlated node signals onto the Laplacian eigenvectors rotates them into orthogonal graph modes. Low-frequency eigenvectors pick out slowly varying components over strongly connected nodes, while higher-frequency eigenvectors pick out sharper contrasts; a raw node trace can mix several such components at once. Two consequences. The cross-series interaction has been absorbed into the basis, so I can model each graph mode with fewer raw node-to-node cross terms in my face. And if the original signals are mixtures of coherent graph components, the eigen-mode series should be less entangled in time than the raw mixed signals. Smoother, less entangled series are individually easier to forecast. So the GFT isn't just a spatial mixer bolted on; it's the step that turns "`N` coupled node series" into graph-mode series with cleaner temporal structure, which is an easier forecasting problem.

But now I've got `N` univariate series sitting in the graph-spectral domain, and they still have temporal structure I need to model. This is the join I was looking for: instead of leaving the GFT output to a time-domain RNN, I model each of these graph-spectral series *with the temporal Fourier transform*. Apply DFT to each eigen-mode series, learn features on the trigonometric coefficients, transform back. Now both transforms are acting on the same representation — GFT put me in the spatial-spectral basis, DFT puts me in the temporal-spectral basis, and the temporal modeling happens *on the graph-spectral series*, not on the raw ones. That's the joint spectral modeling the stacked baselines couldn't do, and it falls out of just refusing to leave the spectral domain between the spatial and temporal steps.

For this to be more than decoration, the DFT step has to be able to do the forecasting itself, not just extract smoothed features for something else to forecast from — worth confirming with the algebra directly. Suppose I have a real series `x_0,…,x_{N-1}` and I want the next value `x_N`. Its DFT is `y_u = (1/N) Σ_{t=0}^{N-1} x_t e^{-i2πut/N}`. Now imagine `x_N` were known and appended; the DFT of the length-`(N+1)` sequence is `ŷ_u = (1/(N+1)) Σ_{t=0}^{N} x_t e^{-i2πut/(N+1)}`, `u=0,…,N`. Compare the two transforms coefficient by coefficient. For `u = 0,…,N-1`,

  `ŷ_u − y_u = Σ_{t=0}^{N-1} x_t ( e^{-i2πut/(N+1)}/(N+1) − e^{-i2πut/N}/N ) + (x_N/(N+1)) e^{-i2πuN/(N+1)}`,

— the first sum is entirely in terms of the *known* history, the second term is the only place the unknown `x_N` enters. And for the new top coefficient `u = N`,

  `ŷ_N = (1/(N+1)) Σ_{t=0}^{N-1} x_t e^{-i2πNt/(N+1)} + (x_N/(N+1)) e^{-i2πN^2/(N+1)}`,

again history plus one `x_N` term. So look at the structure: every one of the `ŷ_u` is (a quantity computable from history) plus (a known phase factor times the single unknown `x_N`). If I knew `ŷ_N`, I could invert that last relation to recover `x_N`,

  `x_N = (N+1) ŷ_N e^{i2πN^2/(N+1)} − Σ_{t=0}^{N-1} x_t e^{i2πN(N-t)/(N+1)}`,

then back-substitute, for `u<N`, as

  `ŷ_u = y_u + Σ_{t=0}^{N-1} x_t ( e^{-i2πut/(N+1)}/(N+1) − e^{-i2πut/N}/N ) + (x_N/(N+1)) e^{-i2πuN/(N+1)}`.

Now the inverse transform is the ordinary length-`N+1` inverse DFT, `x̂_t = Σ_{u=0}^{N} ŷ_u e^{i2πut/(N+1)}`, and because the `{ŷ_u}` were assembled as the DFT coefficients of the extended sequence, it reproduces `x_0,…,x_{N-1}` exactly and hands me `x̂_N` as the forecast. So predicting the next *time-domain* value is equivalent, given the history, to predicting a single *frequency-domain* coefficient `ŷ_N` — the whole forecasting problem reduces to a learnable map `ŷ_N = M(y_0,…,y_{N-1})` in the frequency domain. That's the licence I wanted: a module that operates on DFT coefficients can do forecasting, and unlike a period-detector it doesn't assume the data is periodic — it just learns the frequency-domain map. So the temporal-spectral step is genuinely a forecaster, not decoration.

So what should `M` be — the operator I learn on the frequency coefficients? The output of DFT is complex: a real part `X̂_u^r` and an imaginary part `X̂_u^i`. I'll process them with the same kind of operator but separate parameters, because the real and imaginary parts encode different information (cosine vs sine content) and shouldn't be forced to share weights. For the operator itself I want something that can suppress the frequency components that are noise and keep the ones carrying the sequential pattern — a *gate*. That's exactly a gated linear unit: `GLU(z) = (W₁ z) ⊙ σ(W₂ z)`, where the sigmoid branch learns, per component, how much of the linear branch to let through. At the symbolic level each part runs `M^*(X̂_u^*) = GLU(θ_τ^*(X̂_u^*), θ_τ^*(X̂_u^*)) = θ_τ^*(X̂_u^*) ⊙ σ^*(θ_τ^*(X̂_u^*))` for `* ∈ {r,i}`, with `θ_τ` a small convolution that mixes neighboring frequencies before the gate; in the lean implementation that convolution is dropped and the stacked GLUs do the frequency-domain mixing directly. Recombine into the complex result `M^r + i M^i` and apply the inverse DFT to come back to the time domain. That four-step pipeline — DFT, gated processing on real and imaginary separately, IDFT — is the *Spectral Sequential Cell*, the temporal-spectral learner that sits on each graph-spectral series.

Now wrap the cell back inside the spatial-spectral convolution, because I still owe the graph-convolution filter and the trip back out of the GFT basis. The full block, per output channel `j`, is

  `Z_j = GF^{-1}( Σ_i g_{Θ_ij}(Λ_i) · S( GF(X_i) ) )`,

reading inside-out: GFT each input channel `X_i` into the graph-spectral domain; run the Spe-Seq Cell `S` to learn temporal patterns there; apply the graph-convolution kernel `g_{Θ_ij}(Λ)` — a learnable function of the eigenvalues, which is what "filtering in the graph-spectral basis" means; sum over input channels; and inverse-GFT back to node space. Concatenate the `Z_j` over output channels. That's one StemGNN block — a Spe-Seq Cell *embedded inside* a spectral graph convolution, so the temporal and spatial spectral computations are genuinely fused rather than stacked.

For the output of a block I need to turn `Z` into a forecast, and here I'll borrow a representation that's been shown to work well for deep forecasting: basis expansion. Rather than have the block emit raw numbers, have it emit coefficients `θ` from a fully-connected layer and combine learnable basis vectors `V`, so the output is `Y = V θ`. The basis vectors are shared, learned waveforms; the block only has to say how much of each to use. And I'll give the block *two* such heads, a forecasting head that predicts the future and a backcasting head `B` that reconstructs the block's own input window. Why bother reconstructing the input? Two reasons, and they're the doubly-residual-stacking idea from N-BEATS. First, forcing the block to reconstruct what it consumed *regularizes* its representation — it can't just memorize a forecast, it has to actually model the window. Second, it gives me a clean way to stack blocks: subtract the backcast from the input before passing to the next block, so the second block sees `X − X̂_1`, the part the first block could *not* explain, and only has to learn the residual. The forecasts then sum: `Ŷ = Ŷ_1 + Ŷ_2`. I'll use two blocks — a bilevel residual stack — and feed the summed forecasts through a small `Linear → LeakyReLU → Linear` head. The natural objective has a forecasting error `Σ_t ‖X̂_t − X_t‖²` plus a backcasting reconstruction error `Σ ‖B_{t-i}(X) − X_{t-i}‖²`, but the lean implementation uses the backcast only as the residual passed to the next block; no separate backcast-loss term appears in the forward path.

Before I commit to code, let me cost this, because there's a step here that scares me. The latent correlation layer is `O(N²d)` — fine. The Spe-Seq Cell via FFT is `O(NT log T)` — fine. But the GFT is `U^T X`, and `U` is the eigenvector matrix of the Laplacian, so to do GFT honestly I have to **eigendecompose** `L` — and that's `O(N³)`, every forward pass, on a matrix that *changes* every batch because I'm *learning* `W`. That's not just slow, it's a numerical liability: eigendecomposition of a learned, possibly near-singular Laplacian inside an autograd graph is asking for instability, and backpropagating through `eig` is notoriously fragile. I don't want to forecast through a `torch.eig` call on a freshly-learned matrix tens of thousands of times.

But this is exactly the wall ChebNet hit and climbed years ago, and the escape is sitting right there in the spectral-GCN literature. The graph convolution `U g_θ(Λ) U^T x` only needs the eigendecomposition because the filter `g_θ` is written as an arbitrary function of the eigenvalues. If instead I restrict `g_θ` to a *polynomial* of the Laplacian, the eigenvectors cancel out: `U (Σ_k θ_k Λ^k) U^T = Σ_k θ_k U Λ^k U^T = Σ_k θ_k (U Λ U^T)^k = Σ_k θ_k L^k`, because `U^T U = I`. A polynomial in `Λ` conjugated by `U` is just the same polynomial in `L` — no eigendecomposition needed, only repeated matrix multiplication by `L`. And a degree-`K` polynomial in `L` mixes each node only with its `K`-hop neighborhood, so the filter is automatically *localized*, which is a feature, not a cost. For numerical stability the polynomial basis of choice is Chebyshev, because the `T_k` are orthogonal and the three-term recurrence keeps the powers well-conditioned: `T_0 = I`, `T_1 = L̃`, `T_k = 2 L̃ T_{k-1} − T_{k-2}`. So I replace the explicit `U^T X`/`U X̂` route with a polynomial graph filter — I never form `U` at all. The spectral convolution becomes matmuls against polynomial terms of `L`. In the practical block I keep four channels. The exact Chebyshev stack would be built from `I` and a rescaled `L̃`; the lean recurrence uses the four-channel tensor shape but realizes it as `[0, L, 2L^2, 4L^3 − L]` with the learned normalized Laplacian directly.

Let me lay out the latent correlation layer in code first, since it produces the graph everything else consumes. I feed node-indexed history vectors through the GRU, compute the additive attention from two learned key/query projection vectors, symmetrize, normalize into a Laplacian, and expand the four polynomial channels:

```python
def latent_correlation_layer(self, x):
    # x: [B, K, N]; feed node-indexed history vectors to a GRU
    h, _ = self.GRU(x.permute(2, 0, 1).contiguous())   # [N, B, N]
    h = h.permute(1, 0, 2).contiguous()                # [B, N, N]
    attention = self.self_graph_attention(h)           # [B, N, N] affinities
    attention = torch.mean(attention, dim=0)           # average over batch -> [N, N]
    attention = 0.5 * (attention + attention.T)         # symmetrize: L needs symmetric W
    degree = torch.sum(attention, dim=1)
    degree_l = torch.diag(degree)
    diagonal_degree_hat = torch.diag(1.0 / (torch.sqrt(degree) + 1e-7))  # D^{-1/2}
    laplacian = torch.matmul(diagonal_degree_hat,
                             torch.matmul(degree_l - attention, diagonal_degree_hat))
    # L = D^{-1/2} (D - W) D^{-1/2}  (the normalized Laplacian, built node-wise)
    return self.cheb_polynomial(laplacian)              # [4, N, N]: four polynomial slots

def self_graph_attention(self, inputs):
    # inputs: [B, N, N]; score every ordered pair (i, j) additively, then softmax
    inputs = inputs.permute(0, 2, 1).contiguous()
    bsz, N, _ = inputs.size()
    key = torch.matmul(inputs, self.weight_key)        # [B, N, 1]
    query = torch.matmul(inputs, self.weight_query)    # [B, N, 1]
    data = key.repeat(1, 1, N).view(bsz, N * N, 1) + query.repeat(1, N, 1)  # broadcast sum
    data = self.leakyrelu(data.squeeze(2).view(bsz, N, -1))
    attention = self.dropout(F.softmax(data, dim=2))   # row-normalized affinities
    return attention
```

I symmetrize before taking the degree so the normalized Laplacian is built from the same symmetric `W`; I add `1e-7` under the square root so an isolated low-degree node doesn't divide by zero; I average the attention over the batch so the graph is a single shared structure for the window rather than `B` different graphs. The exact Chebyshev recurrence would begin with `T_0 = I` on a rescaled `L̃`; this lean realization keeps the four polynomial slots but uses a zero first channel and powers of the unrescaled Laplacian:

```python
def cheb_polynomial(self, laplacian):
    N = laplacian.size(0)
    laplacian = laplacian.unsqueeze(0)
    T0 = torch.zeros_like(laplacian)
    T1 = laplacian
    T2 = 2 * torch.matmul(laplacian, T1) - T0          # 2 L T1 - T0
    T3 = 2 * torch.matmul(laplacian, T2) - T1          # 2 L T2 - T1
    return torch.cat([T0, T1, T2, T3], dim=0)          # [4, N, N]
```

Now the block. The Spe-Seq Cell is the DFT, the gated frequency-domain processing on real and imaginary parts separately, and the inverse DFT — exactly the temporal-spectral learner I derived. The `4` here is the four Chebyshev channels coming in; I split the GLU-processed features back into those four channels before the graph-conv kernel multiplies them:

```python
def spe_seq_cell(self, inputs):
    B, _, _, N, L = inputs.size()
    inputs = inputs.view(B, -1, N, L)
    ffted = torch.fft.fft(inputs, dim=-1)              # DFT along time
    real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
    imag = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
    for i in range(3):                                 # gate the frequency features
        real = self.GLUs[i * 2](real)                  # real part: its own params
        imag = self.GLUs[i * 2 + 1](imag)              # imag part: separate params
    real = real.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
    imag = imag.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
    return torch.fft.ifft(torch.complex(real, imag), dim=-1).real   # IDFT, take real part
```

and the block forward ties the polynomial graph filter, the cell, the learnable graph-conv kernel, and the forecast/backcast heads together:

```python
def forward(self, x, graph):
    graph = graph.unsqueeze(1)                         # [4, 1, N, N]
    x = x.unsqueeze(1)                                 # [B, 1, 1, N, L]
    gfted = torch.matmul(graph, x)                     # polynomial graph filtering
    gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
    igfted = torch.matmul(gconv_input, self.weight).sum(dim=1)   # learned graph-conv channel mix
    forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
    forecast = self.forecast_result(forecast_source)             # forecast head (basis expansion)
    if self.layer_idx == 0:
        backcast_short = self.backcast_short_cut(x).squeeze(1)
        backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)  # residual reconstruction
    else:
        backcast_source = None
    return forecast, backcast_source
```

The first block produces a backcast so the second block can take the residual; the second block doesn't need one because nothing follows it. Then the top-level model runs the latent graph once, marches the input through both blocks accumulating forecasts, sums them (the doubly-residual aggregation), and maps to the horizon with the final nonlinear fully-connected head:

```python
def forward(self, inputs, inputs_timestamps):
    graph = self._latent_graph(inputs)                 # [4, N, N]
    x = inputs.unsqueeze(1).transpose(-1, -2)          # [B, 1, N, L]
    results = []
    for i in range(self.num_blocks):
        pred, x = self.stock_block[i](x, graph)        # x becomes the backcast residual
        results.append(pred)
    prediction = sum(results)                          # Y = Y_1 + Y_2
    return self.fc(prediction).transpose(1, 2)         # [B, H, N]
```

That eigendecomposition-free form — learned graph, GFT-then-DFT fused inside one spectral graph convolution, doubly-residual forecast/backcast blocks, Chebyshev polynomial filter in place of the honest eigendecomposition — is the network above.

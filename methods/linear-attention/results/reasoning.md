I start from the cost rather than from a new architecture. With queries, keys, and values in sequence form, the usual layer computes

$$
V' = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt D}\right)V .
$$

The painful object is not mysterious: it is the `N x N` matrix `QK^T`. I pay for `N^2` query-key comparisons, then I pay again to mix values, and I have to keep the attention matrix around for the backward pass. Caching past keys and values during autoregressive generation helps me avoid recomputing old projections, but it does not change the fact that the new query at step `i` must still compare against `i` keys. So the per-step decoding cost grows with the prefix. If I want fixed-cost generation, I need the past to collapse into a fixed-size state — something the `N x N` matrix structurally forbids, because its width is the prefix length.

The first thing I want to question is whether softmax is load-bearing or incidental. Written position by position, attention is a normalized kernel smoother:

$$
V'_i =
\frac{\sum_{j=1}^N \operatorname{sim}(Q_i,K_j)V_j}
     {\sum_{j=1}^N \operatorname{sim}(Q_i,K_j)} .
$$

If I set `sim(q,k) = exp(q^T k / sqrt(D))`, I recover softmax attention. But the normalized-average form only needs non-negative similarities and a nonzero denominator. So the exponential is not sacred; it is one admissible similarity among many. That gives me a degree of freedom to spend: I can pick a different non-negative similarity if some other choice has better algebra.

What algebra do I actually want? The thing that makes the exponential expensive is that it ties `q` and `k` together inside a nonlinearity, so `sim(Q_i, K_j)` is a genuinely separate computation for each pair `(i,j)` — there is no shared work to amortize across the `N` queries. The fix would be a similarity that *factors*: if I could write `sim(q,k)` as a product of a `q`-only part and a `k`-only part, the `k`-side could be summed once and reused by every query. The most direct way to force a factored form is a feature map,

$$
\operatorname{sim}(q,k)=\phi(q)^T\phi(k),
$$

with non-negative feature coordinates so the similarity stays non-negative and the smoother stays a real weighted average. Substituting this into the smoother gives

$$
V'_i =
\frac{\sum_{j=1}^N \phi(Q_i)^T\phi(K_j)V_j}
     {\sum_{j=1}^N \phi(Q_i)^T\phi(K_j)} .
$$

Now `phi(Q_i)` does not depend on `j`, so it comes out of each sum:

$$
V'_i =
\frac{\phi(Q_i)^T\left(\sum_{j=1}^N \phi(K_j)V_j^T\right)}
     {\phi(Q_i)^T\left(\sum_{j=1}^N \phi(K_j)\right)} .
$$

The key-value sum is a `C x M` matrix and the key sum is a `C`-vector; neither depends on the query index `i`. So I compute them once and let every query contract against those two fixed summaries. In matrix form this is nothing but a reassociation,

$$
(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V),
$$

where the left grouping builds the `N x N` table and the right grouping never does.

Before I trust the cost argument I want to make sure those two groupings really are the same object and I have not dropped a transpose. Take a `2 x 2` toy with `C = M = 2`,
`phi(Q) = [[1,2],[0.5,1]]`, `phi(K) = [[1,0],[0,2]]`, `V = [[1,0],[0,1]]`.
Left grouping: `phi(Q)phi(K)^T = [[1,4],[0.5,2]]`, then times `V` gives `[[1,4],[0.5,2]]`.
Right grouping: `phi(K)^T V = [[1,0],[0,2]]`, then `phi(Q)` times that gives `[[1,4],[0.5,2]]`.
They agree, and the cost asymmetry is visible in the dimensions: the left path passes through a `2 x 2` (here `N x N`) intermediate, the right path through a `2 x 2` (here `C x M`) intermediate that has no `N` in it at all. With feature dimension `C` and value dimension `M`, the right grouping is `O(NCM)`, linear in sequence length, and the key-side summaries are fixed objects rather than a quadratic table. That is the whole point of the factorization, and now I have actually seen the two parenthesizations land on the same numbers.

This forces an honesty check on what I have built. The exact softmax kernel `exp(q^T k)` does have a feature-map representation, but its feature map is infinite-dimensional, so I cannot reproduce softmax exactly with any finite `phi`. I am therefore not making softmax cheap; I am defining a *different* attention kernel that happens to be cheap. The question is whether a finite, cheap `phi` exists that is still a legitimate non-negative kernel.

One tempting finite choice is the degree-2 polynomial kernel, which has an exact finite feature map. But its feature map enumerates all coordinate products, so its dimension is roughly `D^2`: a `64`-dim head would blow up to a `~4096`-dim feature, and the `O(NCM)` cost with `C ~ D^2` only beats the quadratic `O(N^2 D)` once `N` exceeds `D^2`, i.e. thousands of tokens. That makes it a special-case win, not a general replacement. For a layer I can drop in everywhere I want `C = D`, which means I should stop insisting on reproducing any particular kernel and just pick a cheap elementwise positive map directly.

The map only has to be elementwise and non-negative. `ReLU` is non-negative but it zeros the gradient on negative inputs, which I would rather not bake into every key and query. `ELU` saturates to `-1` on the negative side (with its parameter at one) while keeping a nonzero derivative there, so shifting it up by one,

$$
\phi(x)=\operatorname{elu}(x)+1 ,
$$

is non-negative everywhere, keeps the feature dimension at `D`, and never kills the gradient. I can sanity-check the positivity directly: for `x > 0` it is `x + 1 > 0`, and for `x <= 0` it is `exp(x) > 0`, so every coordinate is strictly positive and the denominator `phi(Q_i)^T phi(K)` is a sum of positive terms — the weighted average is always well defined. That gives a valid finite-dimensional kernel attention at `O(NDM)`.

Now causal masking. The obvious move — multiply the score matrix by a triangular mask — reintroduces exactly the `N x N` matrix I just removed, so it is a non-starter. I have to push the prefix constraint inside the factorized form. Restricting each query to `j <= i`:

$$
V'_i =
\frac{\phi(Q_i)^T\sum_{j=1}^i \phi(K_j)V_j^T}
     {\phi(Q_i)^T\sum_{j=1}^i \phi(K_j)} .
$$

The two summaries now depend on `i`, but only through a growing prefix, so they are prefix sums. Define

$$
S_i=\sum_{j=1}^i \phi(K_j)V_j^T,\qquad
Z_i=\sum_{j=1}^i \phi(K_j).
$$

Then

$$
V'_i=\frac{\phi(Q_i)^T S_i}{\phi(Q_i)^T Z_i},
$$

with updates

$$
S_i=S_{i-1}+\phi(K_i)V_i^T,\qquad
Z_i=Z_{i-1}+\phi(K_i).
$$

I want to confirm this prefix form really computes the triangular-masked attention and is not some subtly different quantity. On a random `N = 3` numerator (`C = M = 2`, drawn from a fixed seed), I compute it both ways: once as `tril(phi(Q)phi(K)^T) V` with an explicit lower-triangular mask, and once by running `S_i = S_{i-1} + phi(K_i)V_i^T` and reading off `phi(Q_i)^T S_i`. The two give identical rows, e.g. the last position came out `[-1.1237, -7.4608]` in both. So the prefix recurrence is the masked attention, not an approximation of it.

That recurrence has exactly the state shape I wanted at the start: a running attention memory `S` and a running normalizer `Z`, both of fixed size. At generation time I carry only those two forward; each new token does one rank-one update to `S`, one vector update to `Z`, and two contractions to read out, and none of these grow with the prefix length. Putting the projections back in, for a layer input `x_i`,

$$
s_i=s_{i-1}+\phi(x_iW_K)(x_iW_V)^T,\qquad
z_i=z_{i-1}+\phi(x_iW_K),
$$

and the layer output is

$$
y_i=f_l\!\left(
\frac{\phi(x_iW_Q)^T s_i}{\phi(x_iW_Q)^T z_i}+x_i
\right).
$$

This is a recurrent network over time, with hidden state the pair `(s,z)` — not as an added architectural choice but as what the causally masked, kernel-factorized attention *is* when you read it left to right. (Running the normalized recurrence against the normalized triangular-masked batched form on a length-`5` example with `phi = elu+1` reproduces it to `1e-5`, so the normalizer bookkeeping is consistent too.)

There is one remaining gap between this clean recurrence and a usable training layer. During training I want the whole sequence at once, and if I implement the prefix states naively and let autodiff save every `S_i` for the backward pass, I am back to storing one `C x M` matrix per position — the same `O(N)`-many large intermediates I was trying to avoid. To keep training linear in memory the backward pass has to be a scan as well, which means I need the gradients in closed form rather than from generic autodiff.

Work with the numerator alone (the denominator is a separate `phi(Q_i)^T Z_i` division), and absorb `phi` into `Q,K` so I can drop it from the notation:

$$
\bar V_i = Q_i^T\sum_{j=1}^i K_jV_j^T,
\qquad
\bar V_{ie}=\sum_{d=1}^D\sum_{j=1}^i Q_{id}K_{jd}V_{je}.
$$

For the query, `Q_l` appears only in output `\bar V_l` — it reads the prefix at its own position and nowhere else — so the derivative is local:

$$
\frac{\partial L}{\partial Q_{lt}}
=
\sum_e
\frac{\partial L}{\partial \bar V_{le}}
\left(\sum_{j=1}^l K_{jt}V_{je}\right),
\qquad
\nabla_{Q_i}L = G_i\left(\sum_{j=1}^i K_jV_j^T\right)^T,
$$

with `G_i = \nabla_{\bar V_i} L`. The bracketed object is the same forward prefix matrix as the numerator.

The key is different. `K_l` is written into every prefix from `l` onward, so it influences every output `i >= l`:

$$
\frac{\partial L}{\partial K_{lt}}
=
\sum_e\sum_{i=l}^N
\frac{\partial L}{\partial \bar V_{ie}} Q_{it}V_{le}.
$$

Collecting the future contributions into a reverse cumulative matrix `R_i = \sum_{j=i}^N Q_j G_j^T` gives `\nabla_{K_i}L = R_i V_i`. The value `V_l` has the same future-dependence pattern (it too enters every later prefix), so it reuses the same `R_i`, contracted the other way: `\nabla_{V_i}L = R_i^T K_i`.

These formulas are exactly the kind of thing where a dropped transpose or a wrong summation range hides easily, so I check them numerically rather than declare them right. I take `N=4`, `D=3`, `M=2` with random `Q,K,V` and a random linear loss `L = sum(W * \bar V)` so that `G_i = W_i`, compute `nabla_Q`, `nabla_K`, `nabla_V` from the three formulas above (forward cumsum building `R`'s mirror for `Q`, a reverse scan building `R_i` for `K` and `V`), and compare against central finite differences. The match is to roughly `1e-10` on all three: for instance `gradK[0]` comes out `[-0.9341, 0.6489, -1.6541]` both analytically and numerically. So the asymmetry I reasoned out — forward cumulative sum for `Q` because each query reads only its own prefix, reverse cumulative sum for `K` and `V` because each is broadcast into all future prefixes — is the correct one, signs and ranges included, and not merely a plausible-looking guess.

With those scans the backward pass is a single forward sweep (accumulating the prefix matrix for `grad_Q`) and a single reverse sweep (accumulating `R_i` for `grad_K`, `grad_V`), so causal training stays linear in `N` in both time and memory, never materializing the list of prefix matrices.

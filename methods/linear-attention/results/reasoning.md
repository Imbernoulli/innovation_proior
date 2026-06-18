I start from the cost rather than from a new architecture. With queries, keys, and values in sequence form, the usual layer computes

$$
V' = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt D}\right)V .
$$

The painful object is not mysterious: it is the `N x N` matrix `QK^T`. I pay for `N^2` query-key comparisons, then I pay again to mix values, and I have to keep the attention matrix around for the backward pass. Caching past keys and values during autoregressive generation helps me avoid recomputing old projections, but it does not change the fact that the new query at step `i` must still compare against `i` keys. So the per-step decoding cost grows with the prefix. If I want fixed-cost generation, I need the past to collapse into a fixed-size state.

The first useful reframing is to stop treating softmax as the definition of attention. Written position by position, attention is a normalized kernel smoother:

$$
V'_i =
\frac{\sum_{j=1}^N \operatorname{sim}(Q_i,K_j)V_j}
     {\sum_{j=1}^N \operatorname{sim}(Q_i,K_j)} .
$$

If I set `sim(q,k) = exp(q^T k / sqrt(D))`, I recover softmax attention. But the normalized-average form only needs non-negative similarities and a nonzero denominator. That means the exponential is not sacred. I can choose another non-negative similarity if it gives me better algebra.

The algebra I need is factorization. The exponential ties `q` and `k` together inside a nonlinearity, so every pair `(i,j)` is its own computation. Suppose instead I use a feature map and write

$$
\operatorname{sim}(q,k)=\phi(q)^T\phi(k),
$$

with non-negative feature coordinates so the similarity is non-negative. Substituting this into the smoother gives

$$
V'_i =
\frac{\sum_{j=1}^N \phi(Q_i)^T\phi(K_j)V_j}
     {\sum_{j=1}^N \phi(Q_i)^T\phi(K_j)} .
$$

Now the query factor does not depend on `j`. I can pull it outside each sum:

$$
V'_i =
\frac{\phi(Q_i)^T\left(\sum_{j=1}^N \phi(K_j)V_j^T\right)}
     {\phi(Q_i)^T\left(\sum_{j=1}^N \phi(K_j)\right)} .
$$

This is the whole opening. The key-value sum is a `C x M` matrix and the key sum is a `C`-vector; neither depends on the query index `i`. I compute them once, then every query contracts against those two summaries. In vector form I am just changing the parenthesization,

$$
(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V),
$$

and choosing the right parenthesization avoids materializing the `N x N` matrix. If `C` is the feature dimension and `M` the value dimension, the work is linear in sequence length, `O(NCM)`, with fixed key-side summaries rather than a quadratic attention table.

The next wall is the feature map. Exact softmax would need the feature map for the exponential dot-product kernel, and that map is infinite-dimensional. So I cannot honestly say I have made exact softmax cheap. I am defining a different attention kernel. A degree-2 polynomial kernel has a finite feature map, but its feature dimension scales like `D^2`, so it is only attractive when `N` is much larger than `D^2`. For a cheap practical layer I want `C = D`.

An elementwise positive map is enough. ReLU would be non-negative, but it kills the gradient on negative inputs. ELU bottoms out at `-1` when its parameter is one and keeps a nonzero derivative on the negative side, so shifting it gives

$$
\phi(x)=\operatorname{elu}(x)+1 .
$$

Every coordinate is non-negative, the feature dimension stays `D`, and the gradient stays alive for negative coordinates. That gives a valid finite-dimensional kernel attention with `O(NDM)` attention work.

Causal masking is where I have to be careful. If I put the usual triangular mask back onto an explicit score matrix, I have reintroduced the matrix I was trying to remove. So I write the causal version directly:

$$
V'_i =
\frac{\phi(Q_i)^T\sum_{j=1}^i \phi(K_j)V_j^T}
     {\phi(Q_i)^T\sum_{j=1}^i \phi(K_j)} .
$$

The two summaries now depend on `i`, but they are prefix sums. Define

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

So the causal layer has exactly the state shape I was looking for: a running attention memory `S` and a running normalizer memory `Z`. At generation time I only carry those two objects forward. Each new token adds one rank-one update and one vector update, then reads out the current value by two contractions. The state size does not grow with the sequence.

Putting the projections back in makes the recurrent form explicit. For a layer input `x_i`,

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

This is a recurrent network over time. The recurrence is not an extra architectural ornament; it is the causally masked attention computation after the kernel factorization. The hidden state is the pair `(s,z)`.

There is one more trap. During training, if I implement every prefix state naively and let automatic differentiation save all `S_i`, I lose the memory gain. Each `S_i` is a `C x M` matrix, and storing one per position is exactly the wrong thing. I need the backward pass to be a scan too.

For the numerator, I absorb the feature map into `Q` and `K` and write

$$
\bar V_i = Q_i^T\sum_{j=1}^i K_jV_j^T.
$$

For a single component,

$$
\bar V_{ie}=\sum_{d=1}^D\sum_{j=1}^i Q_{id}K_{jd}V_{je}.
$$

The query case is local. `Q_l` appears only in output `\bar V_l`, so

$$
\frac{\partial L}{\partial Q_{lt}}
=
\sum_e
\frac{\partial L}{\partial \bar V_{le}}
\left(\sum_{j=1}^l K_{jt}V_{je}\right).
$$

In vector form, with `G_i = nabla_{\bar V_i} L`,

$$
\nabla_{Q_i}L = G_i\left(\sum_{j=1}^i K_jV_j^T\right)^T.
$$

That is the same forward prefix matrix as the numerator.

The key case is different. `K_l` appears in every later prefix, so it contributes to every output `i >= l`:

$$
\frac{\partial L}{\partial K_{lt}}
=
\sum_e\sum_{i=l}^N
\frac{\partial L}{\partial \bar V_{ie}} Q_{it}V_{le}.
$$

The reverse cumulative matrix is

$$
R_i=\sum_{j=i}^N Q_jG_j^T,
$$

so

$$
\nabla_{K_i}L=R_iV_i.
$$

The value at position `i` has the same future-dependence pattern, so it uses the same `R_i` and contracts the other way:

$$
\nabla_{V_i}L=R_i^T K_i.
$$

The cases now line up cleanly. The query gradient is a forward cumulative sum because each query only reads the prefix available at its own position. The key and value gradients are reverse cumulative sums because each key and value is written into all future prefixes. There are no negative signs or missing constants; the numerator is additive and multilinear, and the only normalizing denominator is handled separately with the final division. With those scans, causal training keeps linear time and does not store the whole list of prefix matrices.

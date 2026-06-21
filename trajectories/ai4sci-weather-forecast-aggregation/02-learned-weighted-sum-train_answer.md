The mean landed at latitude-weighted RMSE 353.50 on z500-3day, 2.6032 on t850-5day, and 3.3991 on wind10m-7day — a working forecaster on all three targets, which confirms the scale-preserving uniform average plugs cleanly into the pretrained ClimaX backbone. But it is a floor, and the shape of those numbers points straight at its one structural defect: the mean weights every variable identically at every location and in every atmospheric state. With $V = 48$ variables — three of them near-inert static fields (land-sea mask, orography, latitude) and many pressure-level fields of very uneven relevance — "count all 48 equally" is almost certainly leaving signal on the table. The 353.50 on z500 is the loudest tell: the 3-day evolution of 500 hPa geopotential is governed by the synoptic dynamical variables (mid-tropospheric geopotential and wind) far more than by the land-sea mask or surface humidity, and the mean dilutes those informative fields by averaging in dozens of comparably-weighted-but-less-relevant ones. The diagnosis is precise: not a scale problem (the mean preserves scale) and not a capacity problem (the backbone is untouched), but that the aggregation has no way to say *this variable matters more than that one*. That is the one degree of freedom I buy back now, as cheaply as possible, before reaching for anything content-dependent.

I propose the **learned weighted sum with softmax normalization**: attach a learnable scalar weight $a_v$ to each variable, normalize the weights to a distribution over the $V$ variables with a softmax, and combine the tokens by that weighted sum,
$$O = \sum_v w_v\, x_v, \qquad w_v = \frac{e^{a_v}}{\sum_j e^{a_j}}.$$
A scalar per variable — not a length-$D$ per-channel vector, not a per-location tensor — is the right resolution because the quantity I am trying to decide is "how much does variable $v$ matter *overall*," which is naturally one number per variable. With $V = 48$ that is 48 parameters total, utterly negligible next to the ViT backbone. This is deliberately the *minimal* step up from the floor: I am not yet making the weighting depend on token contents or location (that is the cross-attention rung above), only letting the global per-variable split move off uniform.

The softmax is not cosmetic — it is forced by what goes wrong without it. If I train $O = \sum_v w_v x_v$ with $w_v$ free real numbers, the map from $(w_1, \dots, w_V)$ to $O$ is linear and homogeneous of degree one in the weights: scale every $w_v$ by $c$ and $O$ scales by $c$. Nothing pins the overall magnitude, so there is an entire ray of settings $c \cdot w$ that the optimizer can drift along. That is exactly the scale-leak the mean's success warned me about — if $\sum_v w_v$ wanders up to 10 or down to 0.1, I have silently multiplied the fused token by 10 or 0.1 and yanked the pretrained stack off the operating point the mean carefully respected, re-creating by the back door the very leak I rejected the bare sum for. And there is no ceiling: an unbounded multiplicative gain in the middle of a deep fine-tuned network is exactly what makes training unstable. The failure points at the cure. I never cared about the *absolute* size of the weights, only the *relative* contribution — a ratio, not a magnitude. So I quotient out the scale: demand the weights sum to one and stay nonnegative, which makes $\sum_v w_v x_v$ a *convex combination* of the variable tokens. It lives in the convex hull of the $x_v$, so it sits on the same single-token scale as any one variable — the property that makes the pretrained backbone happy is *preserved*, not gambled — it cannot introduce an arbitrary gain, and when all $w_v$ are equal it collapses back to the mean.

Softmax is how I parameterize that simplex with free, unconstrained parameters so plain Adam trains it with no projection step. I do not want to carry $w_v \geq 0,\ \sum_v w_v = 1$ as hard constraints during fine-tuning — that means projecting onto the simplex after every step, an ugly special case in the otherwise-vanilla ClimaX training loop. Instead I keep raw parameters $a_v$ living anywhere in $\mathbb{R}^V$ and apply a fixed map: nonnegativity from an arbitrary real comes from the exponential $e^{a_v} > 0$, and sum-to-one from dividing by the total. Together that is the softmax, which does exactly and only what I asked — every $w_v$ strictly positive, summing to one by construction, a valid distribution over the 48 variables read as the model's estimate of how much each should carry. It is differentiable everywhere, and the gradient on $a_v$ couples all the variables through the shared denominator, which is correct: pushing one variable's share up necessarily pushes the others' down, since they compete for a budget that sums to one. The constraint is baked into the functional form, which is why it composes cleanly with the frozen-recipe training loop.

The degenerate case determines where fine-tuning starts, and it is the run I am trying to beat. If all $a_v$ are equal — in particular zero-initialized — then $w_v = e^0 / (V e^0) = 1/V$ for every $v$. Uniform. At initialization the softmax-weighted sum *is* the plain mean over the 48 variables, the exact aggregator that produced 353.50 / 2.6032 / 3.3991. That is the ideal start for a fine-tune-from-pretrained run: I begin from the measured floor — the safe equal-contribution prior that already works with the pretrained backbone — and training perturbs the $a_v$ off uniform only insofar as the latitude-weighted loss rewards it. There is no cold-start risk of the aggregator beginning in a wild corner of the simplex and corrupting the pretrained features; it begins at the centroid and climbs. (Softmax is shift-invariant, so any constant init gives the same uniform start; I zero-initialize the raw weights.)

What I am *not* yet buying is content- and location-dependent mixing. The cross-attention rung above projects each variable into query/key/value and lets a per-location query decide how much to read from each variable — strictly more expressive, since my $w_v$ is a single distribution shared across every location and example whereas attention recomputes the mixing at every token — but it drags in the QKV projection matrices and a per-location attention computation, a real tax on an already-large backbone to answer a question the mean's failure suggests may be largely answerable with a *global* per-variable split. My 48 scalars are the minimal answer to that question; if the global split is most of the signal, I get most of the benefit at essentially none of the cost. I keep only the softmax form — not the ReLU fast-normalized sibling that latency-bound detectors use, and not a per-channel weight — because the task's fine-tuning budget calls for the clean, smooth, exactly-sums-to-one default, zero-initialized so it starts as the measured mean.

I expect this to beat the mean on every target and most clearly on z500-3day, where the floor's largest headroom meets exactly the kind of non-uniform split — upweighting mid-tropospheric geopotential and wind, downweighting the static constants and surface humidity — that should help most. And I already suspect where even this rung runs out: the weighting it learns is *fixed*, so once trained, variable $v$ gets the same share at every grid cell and in every atmospheric state, no matter what its token actually says there. The right weighting at a synoptically active cell surely differs from a quiescent one — so I expect this to beat the mean and then itself be beaten by an aggregator whose weights depend on the token contents, the cross-attention rung directly above.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — step 2: learned weighted sum
class VariableAggregator(nn.Module):
    """Learned weighted sum variable aggregation.

    Learns a scalar weight per variable, applies softmax normalization,
    then computes a weighted sum across variable tokens.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads (unused).
        num_vars (int): Number of input variables V.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # Learnable weight per variable
        self.var_weights = nn.Parameter(torch.zeros(num_vars), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        # Softmax-normalized variable weights
        w = F.softmax(self.var_weights, dim=0)  # V
        w = w.view(1, -1, 1, 1)                # 1, V, 1, 1
        # Weighted sum across variables
        out = (x * w).sum(dim=1)  # B, L, D
        return out
```

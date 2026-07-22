OK, let me think this through from scratch. The thing I keep seeing is that bigger Transformer language models are better, trained on more data they're better, given more compute they're better — but it's all qualitative. Everyone tunes depth and width and number of heads by hand and treats "how big, how long, how much data" as a matter of taste and what fits in memory. I want to know whether there's a *law* underneath this — whether the test loss is a predictable function of the scales I control — because if it is, I can decide the model size and the data budget before I burn the compute, instead of after.

Before any fitting, there's a regularity that decides whether a low-dimensional theory is even possible. If I fix the number of parameters and shuffle the *shape* — make it deeper and thinner, or shallower and wider, change the number of heads, push the aspect ratio around by a factor of forty — how much does the loss move? Barely. A few percent. A six-layer wide model lands within a few percent of a forty-eight-layer one at the same parameter count. So the loss is essentially blind to architecture details and is set by *scale*. If that holds, I can summarize a model by a single number, its size, and ignore the shape — and the corollary is that tuning shape buys percent while scale buys orders of magnitude. That's where I want to spend the effort, so let me go after scale.

First I have to define that single number cleanly, because if I define it sloppily the law will be noisy. What should "model size" be? Count the parameters that actually do the per-token computation — the projection matrices. Per layer: the query/key/value projections, the attention output projection, and the two feed-forward matrices. With width $d_{\text{model}}$, attention width $d_{\text{attn}}$, feed-forward width $d_{\text{ff}}$, that's roughly $2 d_{\text{model}} n_{\text{layer}}(2 d_{\text{attn}} + d_{\text{ff}})$ across $n_{\text{layer}}$ layers. With the conventional shape $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$ this should collapse to something simpler; let me actually substitute. Then $2d_{\text{attn}} + d_{\text{ff}} = 2 d_{\text{model}} + 4 d_{\text{model}} = 6 d_{\text{model}}$, so $2 d_{\text{model}} n_{\text{layer}} \cdot 6 d_{\text{model}} = 12\, n_{\text{layer}} d_{\text{model}}^2$:
$$N \approx 12\, n_{\text{layer}}\, d_{\text{model}}^2.$$
Quick numeric sanity check with a concrete shape, $n_{\text{layer}}=12$, $d_{\text{model}}=768$: the full formula gives $2\cdot768\cdot12\cdot(2\cdot768 + 4\cdot768) = 2\cdot768\cdot12\cdot4608 = 84{,}934{,}656$, and $12\cdot12\cdot768^2 = 144\cdot589{,}824 = 84{,}934{,}656$. Identical, so the reduction is exact under that shape, not just approximate. Now — do I include the token-embedding matrix and the positional embeddings? They have $n_{\text{vocab}} d_{\text{model}}$ and $n_{\text{ctx}} d_{\text{model}}$ parameters. My instinct is to include everything, but those embedding parameters scale with vocabulary, not with the depth-of-computation, and they behave differently. Let me exclude them and call $N$ the *non-embedding* parameter count. The justification is empirical — leaving the embeddings out is what produces the cleaner power laws — so I'm betting on that and will treat $N$ as excluding embeddings.

Compute next, because I'll need it for the allocation question. A forward pass does, per token, about $2N$ FLOPs from the matmuls — the factor of two is the multiply-and-accumulate — plus a context-dependent attention term $2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$. I claimed that term is negligible "when the model is much wider than the context is long"; let me put numbers on it so I know whether I can actually drop it. Take a mid-size model, $n_{\text{layer}}=24$, $d_{\text{model}}=1536$, with $n_{\text{ctx}}=1024$. Then $N = 12\cdot24\cdot1536^2 \approx 6.8\times10^8$, so $2N \approx 1.36\times10^9$, while the context term is $2\cdot24\cdot1024\cdot1536 \approx 7.55\times10^7$. The ratio is about $0.056$ — under 6% of $2N$. The threshold $d_{\text{model}} \gg n_{\text{ctx}}/12$ reads here as $1536 \gg 85$, comfortably true. So in my regime forward $\approx 2N$ to within a few percent. The backward pass is about twice the forward. So total training compute is
$$C \approx 6N \quad\text{FLOPs per token}, \qquad C \approx 6ND \quad\text{over } D \text{ tokens}.$$
Linear in $N$ and $D$ — and that linearity is only honest because I just checked the context term is small; if it weren't, $C$ would carry an $n_{\text{ctx}}$ dependence and the allocation algebra later would not close.

Now the actual laws. Take loss versus model size, with data and compute made abundant so $N$ is the only bottleneck — train a sweep of sizes to convergence and plot test loss against $N$. On a log-log plot the points fall on a straight line over six orders of magnitude. A straight line in log-log *is* a power law: $\log L = -\alpha_N \log N + \text{const}$, i.e.
$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076.$$
I'll write the prefactor as $N_c^{\alpha_N}$ so $N_c$ has units of parameters — a scale at which the loss would be order one. How strong is this dependence, concretely? Doubling $N$ multiplies the loss by $2^{-0.076}$. Let me compute that rather than eyeball it: $2^{-0.076} = e^{-0.076\ln 2} = e^{-0.0527} = 0.9487$, so about a 5.1% reduction per doubling. Small but, on a straight line over six orders of magnitude, utterly reliable. Same procedure for data, now holding $N$ large and varying the dataset size with early stopping (stop when test loss stops falling):
$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad \alpha_D \approx 0.095.$$
And the empirical loss versus compute is also a power law. So each scale, alone, gives a power law. The fit is just a linear regression of $\log L$ on $\log X$; the slope is $-\alpha_X$ and the intercept gives $X_c$ via $\text{intercept} = \alpha_X \log X_c$.

Let me make sure that inversion is actually right before I build everything on top of it, because it's easy to get the sign or the $X_c$ recovery backwards. I'll plant a known law and see if a log-log polyfit gives it back. Take $X_c = 6.4\times10^{13}$, $\alpha = 0.076$, sample $X$ across six orders of magnitude, compute $L=(X_c/X)^\alpha$ exactly, then fit. Running `np.polyfit(log X, log L, 1)` and reading off $\alpha = -\text{slope}$, $X_c = \exp(\text{intercept}/\alpha)$ returns $\alpha = 0.0760$ and $X_c = 6.400\times10^{13}$ — both recovered to the printed digits. So the fit-and-invert recipe is sound; the slope sign and the intercept-to-$X_c$ conversion are correct.

These two laws can't both be the whole story at once, though, because a model isn't trained on infinite data. If I have finite $D$ and keep growing $N$, at some point the model has more capacity than the data can constrain and it overfits — the loss stops following $L(N)$ and bends up. So I need a *joint* law $L(N,D)$ that knows about both, and reduces to each single law in the appropriate limit. Rather than fit some arbitrary surface, let me pin down its form from principles, because the form is what extrapolates.

What must $L(N,D)$ satisfy? If I change the vocabulary or tokenization, the loss just rescales by an overall factor — same model, different units of "token". So whatever form I write must be able to absorb such a rescaling into its constants; that means $N_c$ and $D_c$ are not fundamental, they soak up the vocabulary convention. The limits also have to work: fix $D$ and send $N\to\infty$ (infinite capacity, finite data) and the loss must bottom out at the data-limited value $L(D)$; fix $N$ and send $D\to\infty$ (infinite data, finite capacity) and it must bottom out at the capacity-limited value $L(N)$. The more speculative requirement is the overfitting shape: at large data it should come from the finite-sample variance of the dataset, which scales like $1/D$, so the loss ought to be analytic at $D=\infty$ with a series in *integer* powers of $1/D$.

Let me try to satisfy all three. The vocabulary/limit principles say the two single laws are the two limits, so the joint form is built out of $(N_c/N)^{\text{something}}$ and $(D_c/D)$. The integer-power principle is the sharp one: I want a clean $1/D$ expansion. That pushes me to put $D$ in as a bare $D_c/D$ — first power — rather than as $(D_c/D)^{\alpha_D}$ with a fractional exponent that wouldn't give integer powers. So try
$$L(N,D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D}.$$
Check the limits, and not just by squinting at the exponents — let me put numbers in. Take $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$. Pick $N=10^8$ and push $D$ huge, $D=10^{30}$: the bracket is $(6.4\times10^{13}/10^8)^{0.076/0.103} + 1.8\times10^{13}/10^{30}$, and evaluating the whole thing gives $L=2.76229$, while $L(N)=(N_c/N)^{\alpha_N}$ evaluates to $2.76229$ — equal to all printed digits, so the $D\to\infty$ limit really does land on $L(N)$. The outer exponent $\alpha_D$ and the inner $\alpha_N/\alpha_D$ multiplied back to $\alpha_N$, as the algebra promised. Now the other corner: $N=10^{30}$, $D=10^9$ gives $L=2.74342$, and $L(D)=(D_c/D)^{\alpha_D}$ also gives $2.74342$. Both limits confirmed numerically, not just asserted. And expanding the bracket about $D=\infty$ in $x=D_c/D$ gives $a^{\alpha_D}(1 + \alpha_D x/a + \tfrac{\alpha_D(\alpha_D-1)}{2}x^2/a^2 + \dots)$ with $a=(N_c/N)^{\alpha_N/\alpha_D}$ — a series in $x=D_c/D$, i.e. in integer powers of $1/D$, so the third principle holds too.

A symmetric-looking alternative — $\big[(N_c/N)^{\alpha_N} + (D_c/D)^{\alpha_D}\big]^\beta$ — would also hit the two limits, but expanding it about $D=\infty$ gives powers of $(1/D)^{\alpha_D}$, a *fractional* power, which violates the integer-$1/D$ requirement, and it needs an extra free parameter $\beta$ on top. So the asymmetry between how $N$ and $D$ enter isn't arbitrary; it's forced by demanding the $1/D$ overfitting structure. There's a nice bonus: $L(N)$ at infinite $D$ supplies $\alpha_N, N_c$ and $L(D)$ at infinite $N$ supplies $\alpha_D, D_c$, so every constant in the joint law is already pinned by the two single-variable fits. I'd still re-fit it to the finite-data runs as a check rather than trust that the single-variable constants transfer perfectly to the crossover region; doing so gives $\alpha_N = 0.076$, $\alpha_D = 0.103$, $N_c = 6.4\times10^{13}$, $D_c = 1.8\times10^{13}$, close to the single-variable values, which is the consistency I wanted to see.

This joint law immediately tells me how data must grow with model size to *not* overfit. Overfitting is controlled by the relative size of the two bracket terms, i.e. by the combination $N^{\alpha_N/\alpha_D}/D$. To keep that ratio fixed as I scale up $N$ — to keep the model just barely data-constrained — I need $D \propto N^{\alpha_N/\alpha_D}$. With the fitted exponents that exponent is $0.076/0.103 = 0.738$, so $D \propto N^{0.74}$ roughly: data should grow *sublinearly* with model size. Bigger models need more data, but less than proportionally.

Now the part I actually care about: the budget. I don't get to pick $N$ and $D$ and training time independently; I have a fixed compute budget and I have to spend it well. So I need loss as a function of model size and *training time*, then I'll minimize over the split. For finite training steps $S$ at effectively infinite data, the learning curves — after an early transient — fit
$$L(N, S_{\text{min}}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\text{min}}}\right)^{\alpha_S},$$
with $\alpha_S \approx 0.76$. This one is *additive*, not bracketed. The reason I'd expect addition rather than the bracket here: at infinite data the two effects are independent — a capacity floor $(N_c/N)^{\alpha_N}$ you can't beat with more steps, plus an optimization gap $(S_c/S_{\text{min}})^{\alpha_S}$ that you close by training longer. Independent contributions just add, and there's no overfitting series to respect because data is infinite, so the bracket structure that $L(N,D)$ needed isn't called for.

This also gives a finite-data stopping estimate. If the finite-$D$ curve follows the infinite-data curve until overfitting begins, then the amount I leave on the table by stopping at $S$ is about $(S_c/S_{\text{min}})^{\alpha_S}$. The finite-data optimum cannot keep following the infinite-data curve past the point where that optimization gap is smaller than the finite-data penalty $\Delta L = L(N,D)-L(N,\infty)$. So the stopping point should satisfy
$$S_{\text{stop}}(N,D) \gtrsim \frac{S_c}{\left[L(N,D)-L(N,\infty)\right]^{1/\alpha_S}}.$$
It is a lower bound, not an equality, because the finite-data test loss can slow down before the idealized infinite-data curve reaches that gap; still, it ties early stopping to the same two laws instead of adding a new knob.

What's this $S_{\text{min}}$ rather than raw steps $S$? Because most of my runs were trained at some fixed batch size that isn't the efficient one, and I need to put them on a common footing before I can talk about "optimal" anything. The batch-size physics: there's a critical batch size $B_{\text{crit}}$ such that training with $B$ below it costs essentially no extra *compute* but takes more *steps*, while above it you waste compute. Concretely, to reach a fixed loss, steps and examples trade off as $(S/S_{\text{min}} - 1)(E/E_{\text{min}} - 1) = 1$, and $B_{\text{crit}} \equiv E_{\text{min}}/S_{\text{min}}$. Crucially $B_{\text{crit}}$ depends only on the *loss reached*, not on model size, and it grows as the loss falls — it tracks the gradient noise scale — fitting $B_{\text{crit}}(L) = B_*/L^{1/\alpha_B}$ with $\alpha_B \approx 0.21$. So I define $S_{\text{min}} = S/(1 + B_{\text{crit}}/B)$ as the step count I'd need at $B \gg B_{\text{crit}}$ (minimum steps), and $C_{\text{min}} = C/(1 + B/B_{\text{crit}})$ as the compute I'd use at $B \ll B_{\text{crit}}$ (minimum compute). Standardizing every run to these makes the trends clean and the extrapolation trustworthy; the raw fixed-batch $L(C)$ is contaminated by batch inefficiency.

Now minimize, but be careful: the batch in the efficient compute expression is not a fixed external batch. It is $B_{\text{crit}}(L)$, so it changes with the target loss. If I pretend it is constant, I drop the $\alpha_B$ contribution and get the wrong allocation. The efficient compute is
$$C_{\text{min}} = 6\,N\,B_{\text{crit}}(L)\,S_{\text{min}},\qquad B_{\text{crit}}(L)=B_*/L^{1/\alpha_B}.$$
Let $A=(N_c/N)^{\alpha_N}$ be the capacity term and $T=(S_c/S_{\text{min}})^{\alpha_S}$ be the optimization term, so $L=A+T$. Substituting $S_{\text{min}}=C_{\text{min}}/(6NB_{\text{crit}}(L))$ gives
$$T=\left(6B_*S_c\,\frac{N}{C_{\text{min}}L^{1/\alpha_B}}\right)^{\alpha_S}.$$
At fixed compute I differentiate $L=A+T$ with respect to $N$. The derivative of $T$ contains an implicit $dL/dN$ term from $B_{\text{crit}}(L)$, but exactly at the optimum $dL/dN=0$, so that term drops out. What remains is
$$0=-\frac{\alpha_N}{N}A+\frac{\alpha_S}{N}T,\qquad\text{so}\qquad T=\frac{\alpha_N}{\alpha_S}A.$$
So compute-efficient training stops at a fixed fraction above the converged capacity floor,
$$L=A+T=\left(1+\frac{\alpha_N}{\alpha_S}\right)A.$$
How far above the floor is that, numerically? With $\alpha_N=0.076$, $\alpha_S=0.76$ the ratio $\alpha_N/\alpha_S = 0.10$ exactly, so $L = 1.10\,A$ — the compute-optimal run sits about 10% in loss above the infinite-data converged value for its size. That's a useful gut-check on the whole construction: it says don't train to convergence, stop while you're still roughly 10% short, which matches the lived experience that the last bit of loss is brutally expensive. Now eliminate the variables in terms of $L$: since $A\propto N^{-\alpha_N}$, I have $N\propto L^{-1/\alpha_N}$; since $T\propto S_{\text{min}}^{-\alpha_S}$ and $T$ is proportional to $L$, I have $S_{\text{min}}\propto L^{-1/\alpha_S}$; and by definition $B_{\text{crit}}\propto L^{-1/\alpha_B}$. Multiplying them in $C_{\text{min}}=6NB_{\text{crit}}S_{\text{min}}$ gives
$$C_{\text{min}}\propto L^{-\left(1/\alpha_N+1/\alpha_B+1/\alpha_S\right)}.$$
So the loss-versus-compute exponent is forced to be the reciprocal-of-reciprocals combination:
$$\alpha_C^{\text{min}} = \frac{1}{\,1/\alpha_N + 1/\alpha_S + 1/\alpha_B\,}.$$
The direction with the smallest exponent — the least efficient one, here $\alpha_N$ — has the largest reciprocal and so dominates the sum, and therefore dominates where the compute goes. Let me actually evaluate it rather than leave it symbolic. With the rounded component fits $(\alpha_N,\alpha_S,\alpha_B)=(0.077,0.76,0.21)$: $1/0.077 = 12.99$, $1/0.76 = 1.32$, $1/0.21 = 4.76$, summing to $19.07$, so $\alpha_C^{\text{min}} = 1/19.07 = 0.0525$. The $1/\alpha_N$ term alone is $12.99$ of the $19.07$ — about two-thirds — confirming that the parameter direction dominates. And $0.0525$ sits right next to the directly fit compute exponent of about $0.05$, which is the cross-check I wanted: a number I derived from three *separately* measured exponents lands on a number I measured *independently* off the compute frontier. That agreement is the strongest internal evidence that the whole additive-law-plus-batch-physics picture is self-consistent.

The theoretical allocation among the three directions falls out at the same time:
$$N \propto C_{\text{min}}^{\alpha_C^{\text{min}}/\alpha_N}, \qquad B \propto C_{\text{min}}^{\alpha_C^{\text{min}}/\alpha_B}, \qquad S \propto C_{\text{min}}^{\alpha_C^{\text{min}}/\alpha_S}, \qquad D = B\,S.$$
Plugging the rounded $\alpha_C^{\text{min}}=0.0525$ in: $N$-exponent $0.0525/0.077 = 0.68$, $B$-exponent $0.0525/0.21 = 0.25$, $S$-exponent $0.0525/0.76 = 0.069$, and $D=BS$ has exponent $0.25 + 0.069 = 0.32$. The direct frontier fits put the practical allocation at about $N\sim C^{0.73}$, $B\sim C^{0.24}$, and $S\sim C^{0.03}$, with the step exponent small enough that it may be effectively zero. I should not pretend the rounded-exponent numbers ($0.68/0.25/0.07$) are identical to the direct fits ($0.73/0.24/0.03$) — the $N$ and $S$ exponents differ by a few hundredths — but they tell the same story: as compute grows, most of it should go into a bigger model; batch grows modestly; serial training time grows very slowly; and the amount of data consumed grows far slower than compute. Bigger models are, in effect, more sample-efficient: they reach a given loss in fewer steps and on less data than smaller ones. In practice people train smaller models for longer than this says, but only because of hardware limits, not because it is compute-optimal.

There's a tension lurking if I extrapolate too far. The empirical compute-optimal model size grows around $C^{0.73}$, so one-epoch data use grows only around $C^{0.27}$ — much slower. But the pure data law $L(D)$ says data-limited loss falls only as $D^{-0.095}$. Push both laws out many orders of magnitude and the compute-efficient loss $L(C_{\text{min}})$ would dip below what the slowly growing data could support under $L(D)$. They cross. That cannot be physical, so the laws must break before that crossing — and the crossing point is a natural guess for where this simple scale-only picture stops, perhaps near the maximal performance a Transformer of this kind can reach. The laws also cannot literally continue to zero loss forever, since natural language has nonzero entropy; they must flatten eventually, though I see no sign of flattening yet in the studied range.

Let me write the method as code — the counting, the fits, the stopping estimate, and the allocation. I start with the parameter and compute counts, because every fit depends on these definitions:

```python
import numpy as np


def transformer_param_count(n_layer, d_model, d_ff=None, d_attn=None):
    # Non-embedding parameter count. With the standard shape d_attn = d_ff/4 = d_model
    # this reduces to N ~ 12 * n_layer * d_model**2.
    d_ff = 4 * d_model if d_ff is None else d_ff
    d_attn = d_model if d_attn is None else d_attn
    return 2 * d_model * n_layer * (2 * d_attn + d_ff)


def forward_flops_per_token(N, n_layer, d_model, n_ctx):
    # C_forward ~ 2N + 2 * n_layer * n_ctx * d_model. Training (fwd+bwd) ~ 3x:
    # C ~ 6N per token when d_model >> n_ctx/12.
    return 2 * N + 2 * n_layer * n_ctx * d_model
```

Then I fit a single-variable power law as a straight line in log-log space:

```python
def fit_power_law(X, L):
    # L = (X_c / X) ** alpha  <=>  log L = alpha * (log X_c - log X)
    slope, intercept = np.polyfit(np.log(X), np.log(L), 1)
    alpha = -slope                                   # L falls => negative slope
    X_c = np.exp(intercept / alpha)                  # intercept = alpha * log X_c
    return X_c, alpha
```

For the joint surface, I use the bracket form that reduces to each single law in its limit:

```python
def joint_loss(N, D, params):
    # L(N,D) = [ (N_c/N)^(alpha_N/alpha_D) + D_c/D ]^alpha_D
    # D->inf : -> (N_c/N)^alpha_N = L(N);  N->inf : -> (D_c/D)^alpha_D = L(D)
    alpha_N, alpha_D, N_c, D_c = params
    return ((N_c / N) ** (alpha_N / alpha_D) + D_c / D) ** alpha_D


def fit_joint_loss(runs):
    # runs: array of (N_i, D_i, L_i). Fit the four parameters in log space.
    from scipy.optimize import curve_fit
    runs = np.asarray(runs, dtype=float)
    N, D, L = runs[:, 0], runs[:, 1], runs[:, 2]

    def model(ND, alpha_N, alpha_D, log_Nc, log_Dc):
        Nv, Dv = ND
        params = (alpha_N, alpha_D, np.exp(log_Nc), np.exp(log_Dc))
        return np.log(joint_loss(Nv, Dv, params))

    p0 = (0.076, 0.103, np.log(6.4e13), np.log(1.8e13))
    popt, _ = curve_fit(model, (N, D), np.log(L), p0=p0, maxfev=100000)
    alpha_N, alpha_D, log_Nc, log_Dc = popt
    return alpha_N, alpha_D, np.exp(log_Nc), np.exp(log_Dc)
```

The finite-data stopping estimate is just the optimization-gap equation solved for the step count:

```python
def early_stopping_lower_bound(N, D, params, S_c, alpha_S):
    # S_stop >= S_c / (L(N,D) - L(N,inf))**(1/alpha_S)
    alpha_N, alpha_D, N_c, D_c = params
    finite_data_loss = joint_loss(N, D, params)
    infinite_data_loss = (N_c / N) ** alpha_N
    gap = finite_data_loss - infinite_data_loss
    return S_c / np.maximum(gap, np.finfo(float).tiny) ** (1.0 / alpha_S)
```

For the compute allocation, I keep the function mathematical: it returns the exponents implied by the reciprocal formula for the exponents I pass in, rather than hard-coding the separate empirical frontier fit:

```python
def compute_optimal_exponents(alpha_N, alpha_S, alpha_B):
    # alpha_C^min = 1 / (1/alpha_N + 1/alpha_S + 1/alpha_B)
    alpha_C = 1.0 / (1.0 / alpha_N + 1.0 / alpha_S + 1.0 / alpha_B)
    return {
        "alpha_C_min": alpha_C,          # rounded fits give ~0.052; direct fit ~0.050
        "N_exp": alpha_C / alpha_N,       # theoretical model-size exponent
        "B_exp": alpha_C / alpha_B,       # theoretical critical-batch exponent
        "S_exp": alpha_C / alpha_S,       # theoretical serial-step exponent
        "D_exp": alpha_C / alpha_B + alpha_C / alpha_S,
    }
```

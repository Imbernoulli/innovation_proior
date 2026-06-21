We all know that bigger Transformer language models are better, that more data makes them better, and that more compute makes them better — but that knowledge is qualitative and anecdotal. People pick model size, dataset size, and training duration by intuition and by what happens to fit in memory, and they spend most of their effort tuning depth, width, and the number of attention heads at a fixed size. What I want instead is a *predictive quantitative theory*: an explicit formula for how the test loss $L$ depends on the scales I actually control — the non-embedding parameter count $N$, the dataset size $D$, the training compute $C$, the number of optimization steps $S$, and the batch size $B$ — accurate enough to extrapolate and, above all, accurate enough to answer the one question that matters before any large run: given a fixed compute budget, how should it be split between making the model bigger and training it longer? The existing options do not get there. Earlier work established that generalization error follows power laws in data and model size across several domains, but those trends are reported per-factor, are not specific to Transformer language models, and stop at charting the phenomenon — they are never carried through to a recipe for choosing size and data under a budget. The prevailing shape-search practice is the wrong lever entirely: at fixed $N$ the loss barely moves across very different architectures, so tuning shape buys a few percent while scaling buys orders of magnitude. And fitting any single factor in isolation is incomplete — $L(N)$ ignores finite data and overfitting, $L(D)$ ignores capacity, and an empirical $L(C)$ measured at a fixed, non-critical batch size conflates compute-efficiency with the underlying trend and so extrapolates poorly. Taken one factor at a time, none of these can say how the factors trade off.

What I propose is a set of scaling laws for neural language models — a small family of power laws in $N$, $D$, and $C$, a principled joint law that combines them, and a compute-optimal allocation derived from that joint law. The whole enterprise rests on one empirical regularity that makes a low-dimensional theory possible at all: holding the non-embedding parameter count $N$ fixed and shuffling the *shape* — deeper and thinner, shallower and wider, more or fewer heads, aspect ratio pushed around by a factor of forty — moves the loss by only a few percent. The loss is essentially blind to architecture details and is set by scale, so a single number summarizes a model. To make the laws come out clean I have to define that number carefully. I count only the parameters that do the per-token computation — the query/key/value projections, the attention output projection, and the two feed-forward matrices — giving roughly $2 d_{\text{model}} n_{\text{layer}}(2 d_{\text{attn}} + d_{\text{ff}})$, which under the conventional shape $d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}$ collapses to $N \approx 12\, n_{\text{layer}}\, d_{\text{model}}^2$. I deliberately *exclude* the token-embedding and positional-embedding parameters: they scale with vocabulary rather than with depth-of-computation, behave differently, and including them muddies the laws. Compute follows the same discipline. A forward pass costs about $2N$ FLOPs per token from the matmuls — the factor of two is the multiply-accumulate — plus a context term $2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ that is negligible in my regime where $d_{\text{model}} \gg n_{\text{ctx}}/12$; the backward pass is about twice the forward, so total training compute is $C \approx 6N$ FLOPs per token and $C \approx 6ND$ over $D$ tokens. That linearity is exactly why I dropped the context term and the embeddings.

With $N$ and $C$ defined, the single-factor laws are direct. Sweeping model size with data and compute made abundant, the test loss versus $N$ falls on a straight line in log-log over six orders of magnitude — and a straight line in log-log *is* a power law,

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076,$$

where writing the prefactor as $N_c^{\alpha_N}$ gives $N_c$ units of parameters, a scale at which the loss would be order one. The exponent is small — doubling $N$ multiplies the loss by $2^{-0.076}\approx 0.95$, a 5% gain per doubling — but utterly reliable. The same procedure with $N$ large and the dataset varied (using early stopping) gives $L(D) = (D_c/D)^{\alpha_D}$ with $\alpha_D \approx 0.095$, and loss versus compute is likewise a power law. Each fit is nothing more than a linear regression of $\log L$ on $\log X$: the slope is $-\alpha_X$ and the intercept gives $X_c$.

These two laws cannot both hold at once, because no model is trained on infinite data. With finite $D$, growing $N$ eventually gives the model more capacity than the data can constrain, and the loss stops following $L(N)$ and bends up into overfitting. So I need a *joint* law $L(N,D)$ that knows about both and reduces to each single law in the appropriate limit. Rather than fit an arbitrary surface, I pin down its form from three principles, because the form is what extrapolates. First, changing the vocabulary or tokenization rescales the loss by an overall factor, so the form must absorb such a rescaling into its constants — meaning $N_c$ and $D_c$ are not fundamental. Second, the limits must work: fixing $D$ and sending $N\to\infty$ must bottom out at the data-limited value $L(D)$, and fixing $N$ and sending $D\to\infty$ must bottom out at the capacity-limited value $L(N)$. Third, and sharpest, overfitting at large data comes from the finite-sample variance of the dataset, which scales like $1/D$, so the loss should be analytic at $D=\infty$ with a series in *integer* powers of $1/D$. That third principle is what forces the asymmetry: I put $D$ in as a bare $D_c/D$ — first power — rather than as $(D_c/D)^{\alpha_D}$ with a fractional exponent that would never produce integer powers. The result is

$$L(N,D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D}.$$

The limits check out exactly: as $D\to\infty$ the second term vanishes and the outer exponent $\alpha_D$ multiplies the inner $\alpha_N/\alpha_D$ to give $(N_c/N)^{\alpha_N}=L(N)$; as $N\to\infty$ the first term vanishes and what remains is $(D_c/D)^{\alpha_D}=L(D)$. Expanding the bracket about $D=\infty$ gives the clean integer $1/D$ series. The obvious symmetric alternative $\big[(N_c/N)^{\alpha_N}+(D_c/D)^{\alpha_D}\big]^{\beta}$ would hit both limits too, but it would *not* have a clean $1/D$ expansion and would need an extra free parameter $\beta$ — so the asymmetry is not arbitrary, it is forced by the overfitting structure. Knowing $L(N)$ at infinite $D$ and $L(D)$ at infinite $N$ pins down every constant, with nothing extra to fit; fitting to the finite-data runs gives $\alpha_N = 0.076$, $\alpha_D = 0.103$, $N_c = 6.4\times10^{13}$, $D_c = 1.8\times10^{13}$. Overfitting is controlled by the relative size of the two bracket terms, the combination $N^{\alpha_N/\alpha_D}/D$, so to keep a model just barely data-constrained as it grows, data must scale as $D \propto N^{\alpha_N/\alpha_D} \approx N^{0.74}$ — sublinearly. Bigger models need more data, but less than proportionally.

The budget question needs training *time*, not just data, so I add a law for finite steps at effectively infinite data. After an early transient the learning curves fit an *additive* form,

$$L(N, S_{\text{min}}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\text{min}}}\right)^{\alpha_S}, \qquad \alpha_S \approx 0.76,$$

additive rather than bracketed because at infinite data the two effects are independent: a capacity floor $(N_c/N)^{\alpha_N}$ you cannot beat with more steps, plus an optimization gap $(S_c/S_{\text{min}})^{\alpha_S}$ you close by training longer. This same form gives a finite-data stopping estimate: the finite-$D$ curve tracks the infinite-data curve until overfitting begins, and the amount left on the table by stopping is about $(S_c/S_{\text{min}})^{\alpha_S}$, so the run should not keep following the idealized curve past the point where that optimization gap drops below the finite-data penalty $L(N,D)-L(N,\infty)$, giving

$$S_{\text{stop}}(N,D) \gtrsim \frac{S_c}{\left[L(N,D)-L(N,\infty)\right]^{1/\alpha_S}}.$$

It is a lower bound, not an equality, because the finite-data test loss can slow down before the idealized curve reaches that gap — but it ties early stopping to the same laws rather than adding a new knob. The reason I use $S_{\text{min}}$ rather than raw steps is that most runs were trained at some fixed, non-efficient batch size, and I have to put them on a common footing first. The batch-size physics: there is a critical batch size $B_{\text{crit}}$ below which training costs essentially no extra *compute* but takes more *steps*, and above which compute is wasted; to reach a fixed loss, steps and examples trade off as $(S/S_{\text{min}}-1)(E/E_{\text{min}}-1)=1$ with $B_{\text{crit}}\equiv E_{\text{min}}/S_{\text{min}}$. Crucially $B_{\text{crit}}$ depends only on the loss reached, not on model size, and grows as the loss falls because it tracks the gradient noise scale, fitting $B_{\text{crit}}(L)=B_*/L^{1/\alpha_B}$ with $\alpha_B\approx 0.21$. I define $S_{\text{min}} = S/(1+B_{\text{crit}}/B)$ (the steps needed at $B\gg B_{\text{crit}}$) and $C_{\text{min}} = C/(1+B/B_{\text{crit}})$ (the compute used at $B\ll B_{\text{crit}}$); standardizing every run to these makes the trends clean, where the raw fixed-batch $L(C)$ is contaminated by batch inefficiency.

Now I minimize, carefully. The batch in the efficient-compute expression is not an external constant — it is $B_{\text{crit}}(L)$, which moves with the target loss, and pretending it is constant drops the $\alpha_B$ contribution and gives the wrong allocation. Writing $A=(N_c/N)^{\alpha_N}$ for the capacity term and $T=(S_c/S_{\text{min}})^{\alpha_S}$ for the optimization term, so $L=A+T$, and substituting $S_{\text{min}}=C_{\text{min}}/(6NB_{\text{crit}}(L))$ into the efficient compute $C_{\text{min}}=6NB_{\text{crit}}(L)S_{\text{min}}$ gives $T=\big(6B_*S_c\, N/(C_{\text{min}}L^{1/\alpha_B})\big)^{\alpha_S}$. Differentiating $L=A+T$ with respect to $N$ at fixed compute produces an implicit $dL/dN$ term from $B_{\text{crit}}(L)$, but exactly at the optimum $dL/dN=0$ so that term drops out, leaving $0 = -(\alpha_N/N)A + (\alpha_S/N)T$, i.e. $T = (\alpha_N/\alpha_S)A$. Compute-efficient training therefore stops at a fixed fraction above the converged floor, $L=(1+\alpha_N/\alpha_S)A$, about 10% above the infinite-data converged loss. Eliminating everything in terms of $L$ — $N\propto L^{-1/\alpha_N}$ from $A$, $S_{\text{min}}\propto L^{-1/\alpha_S}$ since $T\propto L$, and $B_{\text{crit}}\propto L^{-1/\alpha_B}$ by definition — and multiplying inside $C_{\text{min}}=6NB_{\text{crit}}S_{\text{min}}$ gives $C_{\text{min}}\propto L^{-(1/\alpha_N+1/\alpha_B+1/\alpha_S)}$, so the loss-versus-compute exponent is forced to be the reciprocal-of-reciprocals combination

$$\alpha_C^{\text{min}} = \frac{1}{\,1/\alpha_N + 1/\alpha_S + 1/\alpha_B\,}.$$

The direction with the smallest exponent — the least efficient one, here $\alpha_N$ — dominates the sum of reciprocals and so dominates where the compute goes. With the rounded fits this gives $\alpha_C^{\text{min}}\approx 0.052$, consistent with the directly fit compute exponent of about $0.05$, and the allocation falls out as $N\propto C_{\text{min}}^{\alpha_C^{\text{min}}/\alpha_N}$, $B\propto C_{\text{min}}^{\alpha_C^{\text{min}}/\alpha_B}$, $S\propto C_{\text{min}}^{\alpha_C^{\text{min}}/\alpha_S}$, and $D=BS$. The component exponents give roughly $N\sim C^{0.68}$, $B\sim C^{0.25}$, $S\sim C^{0.07}$, while the direct frontier fits give about $N\sim C^{0.73}$, $B\sim C^{0.24}$, $S\sim C^{0.03}$; they are not identical but tell the same story — as compute grows, most of it should go into a bigger model, batch grows modestly, serial steps grow very slowly, and data consumed grows far slower than compute. Bigger models are effectively more sample-efficient. One caution on extrapolation: since compute-optimal model size grows around $C^{0.73}$, data use grows only around $C^{0.27}$, but the pure data law $L(D)$ falls only as $D^{-0.095}$ — push both far enough and the compute-efficient loss would dip below what the slowly growing data could support, so the laws must break before that crossing, which is a natural guess for where this scale-only picture stops.

```python
import numpy as np


def transformer_param_count(n_layer, d_model, d_ff=None, d_attn=None):
    d_ff = 4 * d_model if d_ff is None else d_ff
    d_attn = d_model if d_attn is None else d_attn
    return 2 * d_model * n_layer * (2 * d_attn + d_ff)      # ~ 12 n_layer d_model^2


def forward_flops_per_token(N, n_layer, d_model, n_ctx):
    return 2 * N + 2 * n_layer * n_ctx * d_model            # training ~ 3x -> 6N


def fit_power_law(X, L):
    # L = (X_c / X) ** alpha
    slope, intercept = np.polyfit(np.log(X), np.log(L), 1)
    alpha = -slope
    X_c = np.exp(intercept / alpha)
    return X_c, alpha


def joint_loss(N, D, params):
    # L(N,D) = [ (N_c/N)^(alpha_N/alpha_D) + D_c/D ]^alpha_D
    alpha_N, alpha_D, N_c, D_c = params
    return ((N_c / N) ** (alpha_N / alpha_D) + D_c / D) ** alpha_D


def fit_joint_loss(runs):
    from scipy.optimize import curve_fit
    runs = np.asarray(runs, dtype=float)
    N, D, L = runs[:, 0], runs[:, 1], runs[:, 2]

    def model(ND, alpha_N, alpha_D, log_Nc, log_Dc):
        Nv, Dv = ND
        return np.log(joint_loss(Nv, Dv, (alpha_N, alpha_D, np.exp(log_Nc), np.exp(log_Dc))))

    p0 = (0.076, 0.103, np.log(6.4e13), np.log(1.8e13))
    popt, _ = curve_fit(model, (N, D), np.log(L), p0=p0, maxfev=100000)
    a_N, a_D, lNc, lDc = popt
    return a_N, a_D, np.exp(lNc), np.exp(lDc)


def early_stopping_lower_bound(N, D, params, S_c, alpha_S):
    alpha_N, alpha_D, N_c, D_c = params
    finite_data_loss = joint_loss(N, D, params)
    infinite_data_loss = (N_c / N) ** alpha_N
    gap = finite_data_loss - infinite_data_loss
    return S_c / np.maximum(gap, np.finfo(float).tiny) ** (1.0 / alpha_S)


def compute_optimal_exponents(alpha_N, alpha_S, alpha_B):
    alpha_C = 1.0 / (1.0 / alpha_N + 1.0 / alpha_S + 1.0 / alpha_B)
    return {
        "alpha_C_min": alpha_C,
        "N_exp": alpha_C / alpha_N,
        "B_exp": alpha_C / alpha_B,
        "S_exp": alpha_C / alpha_S,
        "D_exp": alpha_C / alpha_B + alpha_C / alpha_S,
    }
```

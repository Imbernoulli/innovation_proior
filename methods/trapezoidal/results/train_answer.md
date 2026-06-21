When I pretrain a large language model, the single hyperparameter that shapes the final model most after the peak learning rate is the *schedule* — how that rate is varied across the run. The default I reach for is a single cosine cycle: warm up to a peak over a few hundred steps, then slide the rate down a cosine curve over the rest of training, ending around a tenth of the peak. The shape itself I understand: a high rate early lets the iterate roam and explore the loss landscape, and the long slow tail at lower and lower rates is what lets the model settle into a good basin instead of rattling around it at the noise floor. The trouble is that the cosine curve is parameterized by its cycle length, and to get the good final model that length has to equal the total number of steps I train for. Hoffmann and the Chinchilla work pinned this down: for a fixed family of models, the best loss at a given token count comes from the cosine whose decay is stretched to exactly that token count. The consequence is what costs me. If I plan a cosine to decay over $130\mathrm{B}$ tokens and read the loss partway through, at some $D' \ll 130\mathrm{B}$, that intermediate loss *overestimates* what a run actually planned to stop at $D'$ would have reached, because most of the loss drop lives in the decay and the decay has not happened yet — the rate is still high and the model is still bouncing. So I cannot mine "quality at length $D'$" from a checkpoint of a longer run. To know how good a model is at several lengths — exactly what scaling-law fits, data-mixture comparisons, and architecture ablations demand — I have to train a separate model from scratch for each length, with a length-matched cosine each time. A family of sizes times several lengths, each from zero, is the bill.

It is also rigid. A cosine run is built to bottom out at its planned end, so its final rate is too low to make further progress; if I want to keep going I either stall or I re-warm the rate, and re-warming spikes the loss and recovers only slowly — the continual-learning literature reports the same, that re-warming hurts and induces forgetting. The competing length-agnostic options do not close the gap either: an inverse-square-root schedule has no distinct final phase to drive the rate low at a chosen stopping point, so its end-of-run loss does not match a length-matched cosine; stepwise and restart schedules re-introduce planning and re-warming spikes; and linear-decay-to-zero, though worst-case optimal, again ramps from immediately after warmup and is parameterized by the total length $T$, so it is no more extendable or reusable than cosine. The root disease is one thing: the schedule's shape is welded to the total length, so I must commit to "how long" before I can even define "how."

I propose the trapezoidal schedule — also called constant-LR-with-cooldown, or warmup-stable-decay (WSD). The starting question is what cosine actually buys, stripped of the cosine-ness: a good trade between a high exploratory rate and a sufficient cool-down to commit, with the cool-down stretched proportionally to the run. If that is all it is, the cosine shape is incidental — what I need is "stay high for a while, then cool down enough," and nothing about staying high requires knowing in advance when I will stop. So I keep the rate flat at a constant value for the main body of training and anneal it only in a short final phase. The flat part is length-agnostic by construction: at every step during it I am in the same exploratory regime, and the moment I decide to stop, I launch the cooldown from wherever I am. The total length never enters the plateau. Concretely, with peak $\eta_{\max}$, total horizon $N$, warmup length $N_{\text{warmup}}$, and cooldown length $N_{\text{decay}}$,

$$
\eta(n) =
\begin{cases}
(n / N_{\text{warmup}})\,\eta_{\max}, & n < N_{\text{warmup}} \\[2pt]
\eta_{\max}, & N_{\text{warmup}} < n \le N - N_{\text{decay}} \\[2pt]
f(n, N, N_{\text{decay}})\,\eta_{\max}, & n > N - N_{\text{decay}},
\end{cases}
$$

with $f$ monotonically decreasing from $1$ to (near) $0$ over the cooldown. The plateau value and the plateau length are decoupled: I can hold the plateau as long as I like, $N_{\text{decay}}$ is a short tail, and $N$ enters only through *when the cooldown starts* — which I get to choose on the fly, even retroactively from a saved plateau checkpoint.

What makes the design choices non-arbitrary, rather than "it seems to work," is the last-iterate analysis for convex Lipschitz objectives — the right yardstick because I ship the last weights, not an average. For SGD $x_{t+1} = x_t - \eta_t g_t$, the suboptimality of the final iterate is bounded by a term whose schedule signature is a double sum, $\tfrac{\gamma}{2}\sum_{k=1}^{T-1} \tfrac{\eta_k}{\sum_{t=k+1}^{T}\eta_t}\cdot \tfrac{1}{\sum_{t=k}^{T}\eta_t}\sum_{t=k}^{T}\eta_t^2\,\mathbb{E}\|g_t\|^2$. Take a *constant* schedule $\eta_t = \eta$ with roughly constant gradient norm $\|g_t\| \approx G$. Then $\sum_{t=k+1}^{T}\eta_t = (T-k)\eta$, $\sum_{t=k}^{T}\eta_t = (T-k+1)\eta$, and $\sum_{t=k}^{T}\eta_t^2 G^2 = (T-k+1)\eta^2 G^2$, so the $k$-th summand collapses to $\eta G^2/(T-k)$, and summing over $k$ gives

$$
\eta\,G^2 \sum_{k=1}^{T-1}\frac{1}{T-k} = \eta\,G^2\,H_{T-1} \approx \eta\,G^2 \ln T.
$$

There is the leak: a constant schedule's last iterate carries an extra $\ln T$ factor — it is worse than the *best* iterate by a logarithmic gap, precisely because at a constant rate the late steps' contributions pile up as a harmonic sum that never decays. This is why the constant plateau alone is not enough and why a cooldown is genuinely needed: letting the late $\eta_k$ shrink toward zero sends the numerator $\eta_k$ for large $k$ to zero and suppresses exactly the late summands that built the harmonic blow-up. So the plateau does exploration and the short tail does the final-iterate cleanup I can trigger whenever I want.

The same theory fixes the *shape* of the tail. The schedule minimizing the worst case of that last-iterate bound is a linear decay of the step size to zero: in the clean Lipschitz setting, $\eta_t = (D/(G\sqrt{T}))(1 - t/T)$ gives

$$
\mathbb{E}\!\left[f(x_T) - f^\star\right] \le \frac{D\,G}{\sqrt{T}},
$$

the optimal $O(1/\sqrt{T})$ rate and now *without* the $\ln T$, exactly because the linear ramp-to-zero zeroes the late $\eta_k$ in the harmonic term computed above; intuitively, a linear decay "emulates iterate averaging," weighting each gradient's net contribution to the final point as a uniform average would, which is what denoises the last iterate. So the canonical cooldown is linear, $f(n) = 1 - (n - (N - N_{\text{decay}}))/N_{\text{decay}}$, running from $1$ at the cooldown's start to $0$ at its end. But worst-case optimality is optimality against the *adversarial* gradient sequence, and a real run's gradient norms have structure, so a problem-adaptive tail can beat linear on the realized sequence. The natural candidate is the $(1-\sqrt{\;})$ cooldown, $f(n) = 1 - \sqrt{(n - (N - N_{\text{decay}}))/N_{\text{decay}}}$. Writing both over cooldown progress $x \in [0,1]$: since $\sqrt{x} \ge x$ on $[0,1]$, $1 - \sqrt{x} \le 1 - x$, so the square-root tail sits *below* linear — at small $x$ (e.g. $x = 0.01 \Rightarrow \sqrt{x} = 0.1$) it has already dropped to $0.9$ while linear is at $0.99$. It front-loads the decay: drop fast first, then crawl to zero along the flattening square-root tail, spending the *end* of the cooldown at very low rates for many steps. That extra low-rate polish is more iterate-averaging at the bottom, and for longer runs with more absolute cooldown steps to fill, I expect $(1-\sqrt{\;})$ to pull ahead, while for short cooldowns the two stay close. Generalizing to $1 - x^a$, an $a$ around $\tfrac{1}{2}$ is the regime to try; very small $a$ drops the rate too far too early and lingers near zero too long. Linear is the safe, theory-blessed default; $(1-\sqrt{\;})$ is the refinement worth having for long cooldowns, so I support both.

The remaining knobs follow from the same pressure. The cooldown should be a short, *tunable* fraction — too short and the late steps cannot shrink smoothly, too long and I am just doing a slow cosine over most of training, throwing away the length-agnostic plateau — so it is exposed as `fract_decay` (commonly $10\%$–$20\%$) rather than baked in, and for longer runs the relevant quantity is enough absolute decay steps, not a sacred percentage. The plateau height must be tuned as its own peak, below a peak that a decaying schedule only briefly visits, because the trapezoid holds the *full* rate through the whole stable phase. The warmup is the mirror of the cooldown — ramp in, hold, ramp out — because the first updates hit a fresh, badly-conditioned model with large unreliable gradients, so I ramp linearly from a small fraction $1/D_{\text{init}}$ of $\eta_{\max}$ (a cosmetic floor, not a real degree of freedom) up to $\eta_{\max}$ to avoid an early blow-up. The final rate's clean theoretical endpoint is zero, but a recipe should keep it an independent dial (`final_lr_factor` / `min_lr`) in case the last phase needs residual adaptability. Two further notes on the picture. A line-interpolation diagnostic checks that the cooldown stays put: walk the straight line in weight space between the pre-cooldown and post-cooldown checkpoints; a loss barrier would mean the tail escaped to a different region, while a smooth drop means it descended into the connected basin the plateau iterate already reached — which is the property that makes branching cooldowns from any plateau checkpoint safe. And weight averaging along the plateau (Polyak averaging, stochastic weight averaging) denoises the returned point and behaves like a decayed-rate schedule, so it is a useful complementary readout, but it changes only the returned weights and does not lower the step sizes driving the dynamics, so the explicit cooldown stays the direct way to suppress the late learning-rate terms. The payoff is the original goal: with cosine the length axis needs $K$ from-scratch runs per model size; with the trapezoid I train each size once at a constant rate for the longest length, save plateau checkpoints, and get each shorter-length stopped model by launching a short cooldown branch — the length axis becomes one long run plus $K$ short tails.

The schedule drops into the loop as a stateless function returning a multiplicative factor on the peak rate, with a warmup branch, a plateau branch, a cooldown branch (linear and `sqrt` are the derived ones; the other simple shapes are exposed by name), and a final-floor branch:

```python
import math


def wsd_schedule(n_iterations, final_lr_factor=0.0, n_warmup=1000,
                 init_div_factor=100, fract_decay=0.1,
                 decay_type="linear"):
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return step / n_warmup + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            x = (step - n_hold) / n_anneal_steps
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (1 - x)
            elif decay_type == "exp":
                return final_lr_factor ** x
            elif decay_type == "cosine":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 + math.cos(math.pi * x)
                ) * 0.5
            elif decay_type == "miror_cosine":
                cosine_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 + math.cos(math.pi * x)
                ) * 0.5
                linear_value = final_lr_factor + (1 - final_lr_factor) * (1 - x)
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (1 - x ** 2)
            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (1 - math.sqrt(x))
            else:
                raise ValueError(
                    "decay type must be one of "
                    "['cosine', 'miror_cosine', 'linear', 'exp', 'square', 'sqrt']"
                )
        else:
            return final_lr_factor

    return schedule


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr,
           frac_decay=0.1, decay_type="linear", warmup_div=100):
    final_lr_factor = min_lr / learning_rate
    schedule = wsd_schedule(
        n_iterations=lr_decay_iters,
        final_lr_factor=final_lr_factor,
        n_warmup=warmup_iters,
        init_div_factor=warmup_div,
        fract_decay=frac_decay,
        decay_type=decay_type,
    )
    return learning_rate * schedule(it)
```

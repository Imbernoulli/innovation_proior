# Thompson Sampling

## The problem

Sequential decision-making under uncertainty: at each step you must choose one of $K$ actions ("arms") whose reward distributions are unknown, observe the reward, and repeat — balancing **exploitation** (play what currently looks best) against **exploration** (gather information that might reveal a better arm). Acting greedily on point estimates can lock onto a confidently-wrong early estimate; exploring uniformly (ε-greedy) wastes trials on arms already known to be bad. The original setting is allocating two medical treatments with unknown success probabilities from meagre, accumulating data.

## The key idea

**Act according to the probability that each action is optimal.** Maintain a Bayesian posterior over the unknown parameters. To choose, **sample one parameter vector from the posterior and play greedily with respect to that sample** (posterior sampling = probability matching). The chance you play arm $k$ then equals the posterior probability that arm $k$ is the optimal arm:

$$ \Pr\big[\text{play } k \mid \text{history}\big] \;=\; \Pr\big[\text{arm } k \text{ is optimal} \mid \text{history}\big]. $$

The randomness comes entirely from posterior uncertainty, so exploration is allocated **in proportion to the chance of being best** (none on hopeless arms, much on uncertain-but-promising ones) and **vanishes automatically** as the posterior concentrates — no exploration schedule or stopping rule is needed.

## Posterior update — Beta-Bernoulli conjugacy

For a Bernoulli arm with unknown success probability $p$, start from the uniform prior $\mathrm{Beta}(1,1)$ on $(0,1)$. By Bayes' rule the posterior after $r$ successes and $s$ failures ($n=r+s$ trials) has density

$$ \frac{(n+1)!}{r!\,s!}\, p^{\,r}(1-p)^{\,s}, $$

which is the $\mathrm{Beta}(r+1,\,s+1)$ density since $\int_0^1 p^{r}(1-p)^{s}\,dp = r!\,s!/(n+1)!$. Conjugacy makes the update trivial: observing one more outcome maps $\mathrm{Beta}(\alpha,\beta) \to \mathrm{Beta}(\alpha+1,\beta)$ on a success and $\to \mathrm{Beta}(\alpha,\beta+1)$ on a failure. The parameters are pseudo-counts; the mean is $\alpha/(\alpha+\beta)$ and the distribution concentrates as $\alpha+\beta$ grows.

The posterior tail equals a binomial cumulative (the regularized incomplete Beta identity):

$$ \int_p^1 \frac{(n+1)!}{r!\,s!}\, t^{r}(1-t)^{s}\,dt \;=\; \sum_{a=0}^{r}\binom{n+1}{a} p^{a}(1-p)^{\,n+1-a}. $$

## Probability one arm beats another — exact two-arm formula

With independent uniform-prior posteriors for arms with counts $(r_1,s_1)$ and $(r_2,s_2)$, $n_i=r_i+s_i$, the probability one unknown success probability exceeds the other has a closed finite form:

$$ \Pr[\,p_2 > p_1\,] \;=\; \frac{\displaystyle\sum_{a=0}^{r_2} \binom{r_1+r_2-a}{r_1}\binom{s_1+s_2+1+a}{s_1}}{\displaystyle\binom{n_1+n_2+2}{\,n_1+1\,}}, \qquad \Pr[\,p_1>p_2\,]=1-\Pr[\,p_2>p_1\,]. $$

Probability-matching with $f(P)=P$ (assign to arm 1 with probability $P=\Pr[p_1>p_2]$) is *identical* to drawing $\tilde p_1\sim\mathrm{Beta}(r_1+1,s_1+1)$, $\tilde p_2\sim\mathrm{Beta}(r_2+1,s_2+1)$ and playing the arm with the larger draw — because $\Pr[\tilde p_1>\tilde p_2]=P$. Per-individual expected sacrifice under $f(P)=P$ is $2PQ\le\tfrac12$ ($Q=1-P$), beating the even-split ($\tfrac12$) whenever any preference exists.

## Algorithm (Beta-Bernoulli)

```python
import random

def thompson_sampling(K, horizon, observe):
    S = [0]*K                     # successes r_k
    F = [0]*K                     # failures  s_k
    for t in range(horizon):
        # sample a plausible mean from each arm's posterior Beta(S+1, F+1)
        theta = [random.betavariate(S[k]+1, F[k]+1) for k in range(K)]
        k = max(range(K), key=lambda i: theta[i])   # play greedily w.r.t. sample
        r = observe(k)                               # 1 = success, 0 = failure
        if r == 1: S[k] += 1                         # conjugate update
        else:      F[k] += 1
    return S, F
```

General form (any prior $p$, action set $\mathcal X$): each step sample $\hat\theta\sim p$, play $x_t=\arg\max_{x\in\mathcal X}\mathbb E_{\hat\theta}[r\mid x]$, then update $p\leftarrow p(\cdot\mid x_t,y_t)$ by Bayes. (Sample the *parameter*, not the predictive 0/1 outcome.)

## Regret guarantee

Stochastic bandit, rewards in $[0,1]$, means $\mu_i$, unique best $\mu^\*=\mu_1$, gaps $\Delta_i=\mu_1-\mu_i$. Expected regret $\mathbb E[R(T)]=\sum_i\Delta_i\,\mathbb E[k_i(T)]$.

**Two-arm bound:** $\displaystyle \mathbb E[R(T)] = O\!\left(\frac{\ln T}{\Delta}+\frac{1}{\Delta^3}\right)$, $\;\Delta=\mu_1-\mu_2$.

**$N$-arm bound:** $\displaystyle \mathbb E[R(T)] \le O\!\left(\Big(\sum_{a=2}^{N}\tfrac{1}{\Delta_a^2}\Big)^{2}\ln T\right)$.

The leading $\ln T$ matches the Lai–Robbins (1985) lower bound $\mathbb E[R(T)]\ge[\sum_i\Delta_i/D(\mu_i\Vert\mu_1)+o(1)]\ln T$, so Thompson Sampling is order-optimal in the horizon.

**Proof sketch (two arms).** Regret $=\Delta\,\mathbb E[k_2(T)]$. Split at $L=24(\ln T)/\Delta^2$ plays of arm 2: the first $L$ give $\le 24\ln T/\Delta$ (the $\ln T/\Delta$ term, from Chernoff concentration of arm 2's posterior to within $\Delta/2$). After $L$ plays, model the stretches between consecutive plays of arm 1 as geometric random variables: with arm 1 at $j$ plays, $s$ successes, the expected wait until $\tilde p_1=\mathrm{Beta}(s+1,j-s+1)$ exceeds threshold $y=\mu_2+\Delta/2$ is $1/(1-F^{\beta}_{s+1,j-s+1}(y))-1=1/F^{B}_{j+1,y}(s)-1$ (Beta-tail = binomial-cdf). Crucially, unlike UCB's deterministic bonus, posterior sampling's exploration is randomized, so arm 2's play probability cannot be bounded uniformly over arm 1's play count; one must track the distribution of arm 1's plays. Summing the truncated geometric means over $j$ in three regimes (small/mid/large $j$, with $D=\mathrm{KL}(y\Vert\mu_1)\sim\Delta^2$ and the middle-regime factor $R^y e^{-Dj}$) gives $\mathbb E[k_2(T)]=O(\ln T/\Delta^2+1/\Delta^4)$; multiplying by $\Delta$ yields the bound.

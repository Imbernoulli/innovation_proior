Logged decision data answers only part of the counterfactual question, and that gap is what makes off-policy evaluation hard. In the one-step contextual-bandit setting each record holds a context $x$, the action $a$ the behavior policy actually took, the propensity with which it took that action, and the single reward observed for that one action; the rewards for every unchosen action are missing. In the sequential setting each record is a full trajectory — states, actions, rewards, next states over a finite horizon — drawn from a behavior policy. In both cases I want the expected reward or discounted return of a *different* target policy, and the logged action frequencies come from the behavior policy, not from the target. The whole problem is correcting that distribution mismatch without collecting fresh target-policy rollouts.

The two standard routes each fail in a complementary way. The direct method learns an outcome model — $q(x,a)=\mathbb{E}[r\mid x,a]$ in a bandit, or an action-value $Q(s,a)$ in an MDP — and averages its predictions under the target policy. Averaging predictions keeps variance low, but the estimate inherits whatever the model gets wrong in the regions the target policy visits, so it is biased exactly where the model is misspecified. Importance sampling does the opposite: it keeps the observed reward and reweights it by $\pi(a\mid x)/p(a\mid x)$ (and, for trajectories, by products of per-step ratios $\rho_t=\pi_{\text{target}}(a_t\mid s_t)/\pi_{\text{behavior}}(a_t\mid s_t)$). It can be unbiased when the propensities are right, but reweighting the raw reward by potentially huge weights — and, in the sequential case, by cumulative products $\rho_1\cdots\rho_H$ whose variance explodes even when each factor has mean one — makes it unstable precisely when the target favors actions the behavior policy rarely tried. What I want is an estimator with an asymmetric, forgiving failure mode: a wrong outcome model should not break validity if the propensities are right, and wrong propensities should not break validity if the outcome model is right.

I propose Doubly Robust Off-Policy Evaluation. The key realization is that this is not a weighted average of the two baselines — averaging two estimators would just average their biases. Instead I want the model to enter as a *baseline* whose target-policy mean I can compute exactly, and I want the logged reward to enter only through an importance-weighted *residual* around that baseline. The useful question is not "which estimator do I trust?" but "what is the smallest random quantity I can safely reweight?" If the model is even moderately good, the residual $r-\hat q(x,a)$ is much smaller than $r$ itself, so I reweight that residual rather than the raw reward, and then add back the model's known target-policy mean to fix its expectation. With $\hat q(x,\pi)=\sum_a \pi(a\mid x)\,\hat q(x,a)$ and importance weight $\mathrm{iw}=\pi(a\mid x)/\hat p(a\mid x)$, the one-step contribution is the estimating equation

$$\hat V_{\text{DR}}=\frac{1}{n}\sum_i\Big[\hat q(x_i,\pi)+\frac{\pi(a_i\mid x_i)}{\hat p_i}\big(r_i-\hat q(x_i,a_i)\big)\Big].$$

The first term is a control variate: $\hat q(x,\pi)$ is the target-policy mean of the model, computable without any reward; the second term is importance sampling applied only to what the baseline missed.

What makes this work is a product-bias identity. Write the model error $\Delta(a,x)=\hat q(x,a)-q(x,a)$ and the propensity error $\delta(a,x)=1-p(a\mid x)/\hat p(a\mid x)$, and treat the nuisance estimates as fixed independently of the evaluation sample. Conditioning on $x$, the residual term replaces $r$ by its mean $q$, and the action expectation of the weighted factual prediction is $(p/\hat p)\,\hat q$. After cancellation the expected contribution is the true target value plus $\sum_a\pi(a\mid x)\,\Delta(a,x)\,\delta(a,x)$, so

$$\mathrm{Bias}(\hat V_{\text{DR}})=\mathbb{E}\Big[\sum_a \pi(a\mid x)\,\Delta(a,x)\,\delta(a,x)\Big],$$

which for a deterministic target is simply $\mathbb{E}[\Delta\,\delta]$. This is the decisive point: the direct method has first-order bias in $\Delta$, importance sampling has first-order bias in the propensity error, but here the bias lives in the *product* $\Delta\delta$. If the reward model is correct, $\Delta=0$; if the propensities are correct, $\delta=0$; either way the product vanishes — one correct nuisance is enough. The same residual structure also tames variance. Dudík–Langford–Li's variance expression shows the dangerous $(1-p)/p$ factor is multiplied by $\Delta^2$, the squared model error, whereas plain IPS multiplies the comparable penalty by the squared reward or target-action value. So even an imperfect model helps, because it shrinks the object being reweighted rather than needing to be exactly right. This is the orthogonality / augmented-inverse-probability insight from the missing-data and causal-inference literature, recast for policy value: the baseline and the weighted factual prediction share an expectation when the propensity is correct, so the model term acts as a correlated control variate subtracted from the noisy IS term, with its known target mean added back.

The sequential case follows by recognizing that every time step is itself a one-step bandit. Step-wise importance sampling can be written recursively as $V_{\text{step}}^{H+1-t}=\rho_t\,(r_t+\gamma\,V_{\text{step}}^{H-t})$: at time $t$ the "context" is $s_t$, the "action" is $a_t$, and the stochastic outcome is the immediate reward plus the discounted future estimate, whose conditional mean is $Q(s_t,a_t)$. So I apply the one-step residual correction at every step. With a fitted $\hat Q$ and $\hat V(s)=\sum_a\pi_1(a\mid s)\,\hat Q(s,a)$, initialize $V_{\text{DR}}^0=0$ and recurse backward:

$$V_{\text{DR}}^{H+1-t}=\hat V(s_t)+\rho_t\big(r_t+\gamma\,V_{\text{DR}}^{H-t}-\hat Q(s_t,a_t)\big).$$

The baseline $\hat V(s_t)$ is the target-policy mean of $\hat Q$; the weighted term corrects the factual residual. Setting $\hat Q\equiv 0$ collapses this back to step-wise importance sampling, while an accurate $\hat Q$ shrinks the residual and drops the variance. Unbiasedness is the one-step cancellation repeated by induction: conditional on $s_t$, $\mathbb{E}_{a_t\sim\pi_0}[\rho_t\,\hat Q(s_t,a_t)]=\hat V(s_t)$, so the model term has zero net effect on the expectation, the weighted reward-plus-future term transports the behavior sample to the target distribution, and the future term is unbiased by hypothesis. The variance recursion (Jiang–Li) is honest about the limits: it splits into a transition term, an action-stochasticity term $\mathbb{E}[\mathrm{Var}(\rho_t\,\Delta(s_t,a_t)\mid s_t)]$, a reward-noise term, and a discounted future-variance term, and only the action term depends on $\Delta$. A better value model shrinks that one term; it cannot erase transition or reward noise. Indeed, with a perfect $\hat Q=Q$ the reducible action-mismatch variance disappears and what remains is the intrinsic floor — Jiang–Li's lower-bound argument shows the perfect-model estimator reaches the Cramér–Rao bound for unbiased off-policy estimators in tree MDPs. The one condition I cannot skip is fold independence: the cancellation identities $\mathbb{E}[\rho\,\hat q]=\hat q(\pi)$ and $\mathbb{E}[\rho\,\hat Q]=\hat V$ hold only when the nuisance models are treated as fixed while expectations are taken over the logged action, so $\hat q$, $\hat V$, and any estimated propensities must be fit on data independent of the evaluation samples, via sample splitting or cross-fitting that still uses all the data across folds.

```python
import numpy as np

def doubly_robust_bandit(rewards, actions, pscore, contexts, pi_target, q_hat):
    vals = []
    for r, a, p, x in zip(rewards, actions, pscore, contexts):
        pi_x = pi_target(x)
        q_x = q_hat(x)
        baseline = float(np.dot(pi_x, q_x))
        iw = pi_x[a] / p
        vals.append(baseline + iw * (r - q_x[a]))
    return float(np.mean(vals))

def doubly_robust_sequential(trajectory, gamma, pi_target, pi_behavior, q_hat, v_hat):
    value = 0.0
    for s, a, r in reversed(trajectory):
        rho = pi_target(s)[a] / pi_behavior(s)[a]
        value = v_hat(s) + rho * (r + gamma * value - q_hat(s, a))
    return value
```

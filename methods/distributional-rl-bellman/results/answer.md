# The distributional Bellman operator and its contraction theory

## Problem

Value-based reinforcement learning models only the *expected* return $Q^\pi(x,a)=\mathbb{E}[\sum_t\gamma^t R_t]$, which obeys Bellman's equation and a sup-norm $\gamma$-contraction. This discards the *shape* of the random return — multimodality and the intrinsic randomness from stochastic rewards, dynamics, and policy. The goal is a Bellman equation directly on the **distribution** of the return, with the same well-posedness (unique fixed point, geometric convergence), and an honest account of the control case.

## Setup

MDP $(\mathcal{X},\mathcal{A},R,P,\gamma)$, $\gamma\in[0,1)$, $R$ treated as a random variable. The **random return** and **value distribution**:
$$Z^\pi(x,a)=\sum_{t\ge0}\gamma^t R(x_t,a_t),\quad x_0{=}x,\ a_0{=}a,\qquad Q^\pi=\mathbb{E}Z^\pi.$$
$\mathcal{Z}$ = value distributions with bounded moments. Transition operator and **distributional Bellman operator**:
$$P^\pi Z(x,a)\overset{D}{:=}Z(X',A'),\quad X'\sim P(\cdot|x,a),\ A'\sim\pi(\cdot|X');\qquad \mathcal{T}^\pi Z(x,a)\overset{D}{:=}R(x,a)+\gamma\,P^\pi Z(x,a),$$
with $R$, $(X',A')$, and $Z(X',A')$ independent.

## Metric

$p$-Wasserstein (Mallows) distance between cdfs, via the quantile coupling:
$$d_p(F,G)=\inf_{U,V}\|U-V\|_p=\|F^{-1}(\mathcal{U})-G^{-1}(\mathcal{U})\|_p=\Big(\int_0^1|F^{-1}(u)-G^{-1}(u)|^p\,du\Big)^{1/p}\ (p<\infty).$$
Properties ($a$ scalar, $A$ a r.v. independent of $U,V$): (P1) $d_p(aU,aV)\le|a|d_p(U,V)$; (P2) $d_p(A{+}U,A{+}V)\le d_p(U,V)$; (P3) $d_p(AU,AV)\le\|A\|_p d_p(U,V)$. **Maximal Wasserstein** over value distributions: $\overline d_p(Z_1,Z_2):=\sup_{x,a}d_p(Z_1(x,a),Z_2(x,a))$, a metric.

## Theorem 1 (policy evaluation: contraction)

$\mathcal{T}^\pi:\mathcal{Z}\to\mathcal{Z}$ is a $\gamma$-contraction in $\overline d_p$, for $1\le p\le\infty$.

**Proof.** At fixed $(x,a)$, the shared reward drops out by P2, $\gamma$ comes out by P1, and $P^\pi$ is controlled by coupling the same successor draw on both sides, then using an optimal conditional coupling for that successor:
$$d_p(\mathcal{T}^\pi Z_1(x,a),\mathcal{T}^\pi Z_2(x,a))=d_p(R+\gamma P^\pi Z_1,\,R+\gamma P^\pi Z_2)\le\gamma\,d_p(P^\pi Z_1,P^\pi Z_2)\le\gamma\sup_{x',a'}d_p(Z_1(x',a'),Z_2(x',a')).$$
Taking $\sup_{x,a}$: $\overline d_p(\mathcal{T}^\pi Z_1,\mathcal{T}^\pi Z_2)\le\gamma\,\overline d_p(Z_1,Z_2)$. $\square$

By Banach, $\mathcal{T}^\pi$ has a unique fixed point; by inspection it is $Z^\pi$. With bounded moments, $Z_{k+1}=\mathcal{T}^\pi Z_k\to Z^\pi$ in $\overline d_p$ and all moments converge geometrically. The contraction modulus follows from the contraction-method machinery for distributional recursive equations (Rösler 1991/1992).

**The metric is essential.** $\mathcal{T}^\pi$ is *not* a contraction in total variation (Chung & Sobel 1987), KL, or Kolmogorov distance: the discount $\gamma$ is a horizontal transport on the value axis, which Wasserstein measures and likelihood metrics do not. **Variance** has a $\gamma^2$ propagation term: by the law of total variance,
$$\mathrm{Var}(\mathcal{T}^\pi Z)=\mathrm{Var}(R)+\gamma^2\mathbb{E}_{P,\pi}\mathrm{Var}(Z(X',A'))+\gamma^2\mathrm{Var}_{P,\pi}(\mathbb{E}Z(X',A')) .$$
For value distributions with matching successor means, the full variance difference satisfies
$$\|\mathrm{Var}(\mathcal{T}^\pi Z_1)-\mathrm{Var}(\mathcal{T}^\pi Z_2)\|_\infty\le\gamma^2\|\mathrm{Var}Z_1-\mathrm{Var}Z_2\|_\infty.$$
More generally, after subtracting the mean-dependent transition term, the conditional-variance component has this $\gamma^2$ contraction.

## Control: the subtlety

The optimal value distributions $\mathcal{Z}^*=\{Z^{\pi^*}:\pi^*\in\Pi^*\}$ are generally *many* (a distribution with mean $Q^*$ need not be optimal). A distributional optimality operator is a *selection rule* $\mathcal{T}Z=\mathcal{T}^\pi Z$ for some $\pi\in\mathcal{G}_Z$ (greedy w.r.t. $\mathbb{E}Z$).

**Mean contracts:** $\|\mathbb{E}\mathcal{T}Z_1-\mathbb{E}\mathcal{T}Z_2\|_\infty\le\gamma\|\mathbb{E}Z_1-\mathbb{E}Z_2\|_\infty$ (since $\mathbb{E}\mathcal{T}_D Z=\mathcal{T}_E\mathbb{E}Z$), so $\mathbb{E}Z_k\to Q^*$ geometrically.

**Proposition (not a contraction).** $\mathcal{T}$ is a contraction in no distribution metric. *Counterexample* (undiscounted): $x_1\to x_2$; at $x_2$, $a_1$ gives $0$, $a_2$ (optimal) gives $1{+}\epsilon$ or $-1{+}\epsilon$ each w.p. $\tfrac12$; both terminal. Unique $Z^*$. Take $Z=Z^*$ except $Z(x_2,a_2)=-\epsilon\pm1$: then $\overline d_1(Z,Z^*)=2\epsilon$. But $\mathbb{E}Z(x_2,a_2)=-\epsilon<0$ flips greedy to $a_1$, so $\mathcal{T}Z(x_1)=\delta_0$ while $\mathcal{T}Z^*(x_1)=\epsilon\pm1$, giving
$$\overline d_1(\mathcal{T}Z,\mathcal{T}Z^*)=\tfrac12|1-\epsilon|+\tfrac12|1+\epsilon|>2\epsilon$$
for small $\epsilon$ — an expansion. $\square$ The cause: greedy selection is discontinuous — a vanishing mean change flips the argmax and swaps in a wholly different distribution (bimodal $\leftrightarrow$ Dirac). Consequently $\mathcal{T}$ may also have no fixed point (limit cycles under tie-breaking) and, even when it does, iterates may converge only to the **nonstationary optimal value distributions** $\mathcal{Z}^{**}$.

**Theorem 2 (control convergence).** With $\mathcal{A}$ finite and $\mathcal{X}$ measurable, for finite $p$,
$$\lim_{k\to\infty}\inf_{Z^{**}\in\mathcal{Z}^{**}}d_p(Z_k(x,a),Z^{**}(x,a))=0\quad\forall x,a,$$
uniformly if $\mathcal{X}$ is finite. With a total order $\prec$ on $\Pi^*$ used to break greedy ties, $\mathcal{T}=\mathcal{T}^{\pi^*}$ has a unique fixed point $Z^*\in\mathcal{Z}^*$.

*Proof idea.* Let $B$ be the return-range diameter and $\epsilon_k=\gamma^k B$; mean contraction $\Rightarrow$ on states $\mathcal{X}_k$ with mean-gap $>2\epsilon_k$ the greedy policy is optimal. Recursive sets $\mathcal{X}_{k,i}$ require the bad-successor probability to be at most $\delta^p$, so the bad-successor indicator has $L_p$ norm at most $\delta$, and satisfy $\mathcal{X}_{k,i}\uparrow\mathcal{X}$. The **partition lemma** ($d_p(U,V)\le\sum_i d_p(A_iU,A_iV)$ for a partition $\{A_i\}$ of $\Omega$) splits each backup into a solved term (γ-contracted) and an unsolved term (bounded by $\gamma\delta B$ via P3); inducting on $i$ gives $d_p(W_{k+i}(x),W^*(x))\le\gamma^i B+\delta B/(1-\gamma)\to0$.

## Why categorical/quantile parameterization

To implement $\mathcal{T}$ with a finite model, use a discrete distribution on a fixed grid $\{z_i=V_{\min}+i\Delta z\}_{i=0}^{N-1}$, $\Delta z=\frac{V_{\max}-V_{\min}}{N-1}$, $p_i=\mathrm{softmax}(\theta_i)$. Two obstacles:

1. **Disjoint supports.** $\mathcal{T}Z_\theta$ lives on $\{r+\gamma z_i\}\ne\{z_i\}$.
2. **Wasserstein is unminimizable from samples.** For a mixture $P=P_I$ and any $Q$: $d_p(P,Q)\le\mathbb{E}_{i}d_p(P_i,Q)$ with strict inequality and $\nabla_Q d_p(P_I,Q)\ne\mathbb{E}_i\nabla_Q d_p(P_i,Q)$. (E.g. $P=\mathrm{Bern}(\tfrac12)$ on $\{0,1\}$, $Q$ puts $p$ on $0$: true $d_1=|p-\tfrac12|$, but sampled loss $\equiv\tfrac12$ — flat.)

**Resolution — projected categorical operator.** Backup each atom $\widehat{\mathcal{T}}z_j=[r+\gamma z_j]_{V_{\min}}^{V_{\max}}$ and distribute $p_j(x',\pi(x'))$ to its two nearest grid points:
$$(\Phi\widehat{\mathcal{T}}Z_\theta(x,a))_i=\sum_{j=0}^{N-1}\Big[1-\frac{|[\widehat{\mathcal{T}}z_j]_{V_{\min}}^{V_{\max}}-z_i|}{\Delta z}\Big]_0^1\,p_j(x',\pi(x')),$$
then minimize the cross-entropy of $D_{\mathrm{KL}}(\Phi\widehat{\mathcal{T}}Z_{\tilde\theta}(x,a)\,\|\,Z_\theta(x,a))$ — the Bellman update as multiclass classification, trainable by SGD in $O(N)$. This is principled: the projected operator $\Phi\mathcal{T}^\pi$ is a $\sqrt{\gamma}$-contraction in the **Cramér distance** $\ell_2(F,G)=(\int(F{-}G)^2)^{1/2}$, equivalently a $\gamma$-contraction in squared Cramér distance, because $\Phi$ is a Cramér non-expansion onto the grid. The same pressure also motivates inverse-cdf/quantile coordinates: fix probabilities, learn support locations, and align the representation with Wasserstein geometry.

## Projected categorical backup

```python
import numpy as np

V_MIN, V_MAX, N = -10.0, 10.0, 51
z  = np.linspace(V_MIN, V_MAX, N)            # canonical returns {z_i}
dz = (V_MAX - V_MIN) / (N - 1)

def project_distribution(rewards, terminals, next_probs, gamma):
    """Phi T-hat Z, Eq. (cat_proj): apply r + gamma z_j to each atom, clip to
    [V_min, V_max], and smear p_j(x', pi(x')) onto the two nearest grid points
    by linear interpolation.  Returns the categorical target on {z_i}."""
    B = rewards.shape[0]
    m = np.zeros((B, N))
    for b in range(B):
        g = gamma * (1.0 - terminals[b])                  # terminal -> gamma_t = 0
        for j in range(N):
            Tz   = min(V_MAX, max(V_MIN, rewards[b] + g * z[j]))
            bj   = (Tz - V_MIN) / dz                       # position in [0, N-1]
            l, u = int(np.floor(bj)), int(np.ceil(bj))
            if l == u:
                m[b, l] += next_probs[b, j]
            else:
                m[b, l] += next_probs[b, j] * (u - bj)
                m[b, u] += next_probs[b, j] * (bj - l)
    return m

def greedy_next_action(next_logits):                       # greedy w.r.t. E[Z]
    p = _softmax(next_logits, -1)                          # (B, A, N)
    return (p * z).sum(-1).argmax(-1)                      # (B,)

def categorical_loss(online_logits_chosen, target_dist):   # cross-entropy of KL
    logp = online_logits_chosen - _logsumexp(online_logits_chosen, -1, keepdims=True)
    return -(target_dist * logp).sum(-1).mean()

def _softmax(x, ax):  e = np.exp(x - x.max(ax, keepdims=True)); return e / e.sum(ax, keepdims=True)
def _logsumexp(x, ax, keepdims=False):
    m = x.max(ax, keepdims=True); s = np.log(np.exp(x - m).sum(ax, keepdims=True)) + m
    return s if keepdims else np.squeeze(s, ax)
```

## Causal chain

Keep the return's distribution → write $Z\overset{D}{=}R+\gamma Z'$ → prove $\mathcal{T}^\pi$ is a $\gamma$-contraction in maximal Wasserstein $\overline d_p$ (metric forced by the discount being a horizontal transport) → unique fixed point $Z^\pi$, geometric convergence, variance propagates at $\gamma^2$ once the mean term is fixed → **control breaks it**: mean contracts but greedy selection is discontinuous, so $\mathcal{T}$ contracts in no distribution metric and only reaches the nonstationary optimal set → represent on a fixed atom grid, abandon the (sample-unminimizable) Wasserstein loss, **project** onto the grid and minimize cross-entropy, principled via the $\sqrt{\gamma}$ Cramér contraction of the projected operator.

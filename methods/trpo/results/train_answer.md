We want to train large, nonlinear stochastic policies — neural networks with tens of thousands of parameters — for sequential decision making, and to do it in a way that makes *reliable, monotonic* progress with little hyperparameter tuning. On paper this should favor gradient methods: with a real gradient oracle their sample complexity need not even depend on dimension, whereas black-box search scales with the parameter count, and supervised deep learning just showed that plain stochastic gradient descent rides millions of parameters without drama. Yet in policy optimization the embarrassing fact is that derivative-free methods — the cross-entropy method, CMA-ES — are routinely competitive with or better than gradient methods on hard problems like Tetris and locomotion. Vanilla policy gradient ascends the return with the score-function estimate $\nabla_\theta \eta = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)\,A_\pi(s,a)]$ and takes $\theta \leftarrow \theta + \alpha \nabla_\theta \eta$, but the entire game collapses onto the choice of $\alpha$: too small and we crawl through the noise of a high-variance estimate, exactly the regime where random search keeps pace; too large and a single update can knock the policy off a cliff, and because the next batch of data is gathered *under the wrecked policy*, there may be no recovery. The natural policy gradient improves the direction by preconditioning with the inverse Fisher matrix, $F^{-1}g$, so the step is steepest ascent in distribution space rather than coordinate space — but it commits to a *fixed* step or penalty, which over- or under-shoots from task to task, and forming or inverting $F$ is infeasible for a large network. Conservative policy iteration is the one method with an explicit monotonic-improvement guarantee, but it holds only for unwieldy mixture policies nobody parameterizes with a net. The real defect underneath all of this is that we are measuring the step in the wrong space: $\alpha\nabla\eta$ moves $\theta$ by some Euclidean amount, but $\theta$ are the weights of a network whose output is a *distribution*, and Euclidean distance in weight space has no fixed relationship to how much $\pi(\cdot|s)$ actually changes. The right question is not "what step size in $\theta$" but "how far should the *policy* move per update, measured in distribution space, so that the true return reliably goes up?"

I propose Trust Region Policy Optimization (TRPO): at each iteration, maximize a surrogate for the return subject to a hard bound on how far the policy is allowed to move in KL divergence. The starting point is an exact identity relating a new policy $\tilde\pi$ to the current one $\pi$,
$$\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tau\sim\tilde\pi}\!\left[\sum_t \gamma^t A_\pi(s_t,a_t)\right],$$
where the advantage $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$ is measured under the *old* policy but the trajectory is rolled out under the *new* one. This is exact: writing $A_\pi(s,a) = \mathbb{E}_{s'}[r(s) + \gamma V_\pi(s') - V_\pi(s)]$ and taking the expectation over a $\tilde\pi$-trajectory, the $V$ terms telescope to $-V_\pi(s_0)$ (the tail $\gamma^{T+1}V(s_{T+1})\to 0$ since $\gamma<1$), leaving $-\eta(\pi)+\eta(\tilde\pi)$. Collected by state, $\eta(\tilde\pi) = \eta(\pi) + \sum_s \rho_{\tilde\pi}(s)\sum_a \tilde\pi(a|s)A_\pi(s,a)$ with $\rho_\pi(s)=\sum_t\gamma^t P(s_t=s)$ — but the weighting is the *new* policy's visitation $\rho_{\tilde\pi}$, which depends on the very thing we are solving for. The tractable move is to freeze it: define the surrogate
$$L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi(s)\sum_a \tilde\pi(a|s)A_\pi(s,a),$$
which matches $\eta$ in value and in gradient at $\tilde\pi=\pi$. The gradient match is the load-bearing fact: differentiating the exact objective at $\theta_0$ produces a policy-gradient term plus a $\nabla\rho$ term multiplied by $\sum_a\pi_{\theta_0}(a|s)A_{\theta_0}(s,a)$, which is *zero* at every state because the advantage is centered under the old policy, so the visitation-derivative term drops and $\nabla_\theta L_{\theta_0}|_{\theta_0}=\nabla_\theta\eta|_{\theta_0}$. So a sufficiently small improving step on $L$ improves $\eta$ — but we are back to "sufficiently small" with no quantitative leash.

The leash comes from a coupling argument that converts the gap $|\eta(\tilde\pi)-L_\pi(\tilde\pi)|$ into something quadratic in policy distance. The only difference between the two quantities is whether the states are drawn under $\tilde\pi$ or $\pi$, since with $\bar A(s)=\mathbb{E}_{a\sim\tilde\pi}[A_\pi(s,a)]$ both read $\eta(\pi)+\mathbb{E}_\tau[\sum_t\gamma^t\bar A(s_t)]$ under their respective trajectory law. Call $(\pi,\tilde\pi)$ an $\alpha$-coupled pair if there is a joint law over $(a,\tilde a)|s$ with the right marginals and $P(a\neq\tilde a|s)\le\alpha$ — the general version of the mixture's "act like the old policy with probability $1-\alpha$." Because $\mathbb{E}_{a\sim\pi}[A_\pi(s,a)]=0$, we can write $\bar A(s)=\mathbb{E}_{(a,\tilde a)}[A(s,\tilde a)-A(s,a)]$, which only the disagreement events contribute to, giving $|\bar A(s)|\le 2\alpha\varepsilon$ with $\varepsilon=\max_{s,a}|A_\pi(s,a)|$. Coupling the *trajectories* with a shared seed and conditioning on the number of prior disagreements $n_t$: when $n_t=0$ the two trajectories are identical so those terms cancel, leaving $|\mathbb{E}_{s_t\sim\tilde\pi}[\bar A]-\mathbb{E}_{s_t\sim\pi}[\bar A]|\le 4\varepsilon\alpha\,(1-(1-\alpha)^t)$ using $P(n_t>0)\le 1-(1-\alpha)^t$. Summing the geometric series, $\sum_t\gamma^t 4\varepsilon\alpha(1-(1-\alpha)^t)=4\varepsilon\alpha\big(\tfrac{1}{1-\gamma}-\tfrac{1}{1-\gamma(1-\alpha)}\big)=\tfrac{4\varepsilon\gamma\alpha^2}{(1-\gamma)(1-\gamma(1-\alpha))}\le \tfrac{4\varepsilon\gamma}{(1-\gamma)^2}\alpha^2$. The maximal-coupling lemma lets $\alpha=D_{TV}^{\max}(\pi,\tilde\pi)$, and Pinsker's inequality $D_{TV}^2\le D_{KL}$ pushes it into KL, yielding
$$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\,D_{KL}^{\max}(\pi,\tilde\pi), \qquad C = \frac{4\varepsilon\gamma}{(1-\gamma)^2}.$$
This is a genuine minorize-maximize scheme: setting $M_i(\pi)=L_{\pi_i}(\pi)-C\,D_{KL}^{\max}(\pi_i,\pi)$, the bound gives $\eta\ge M_i$ everywhere with equality at $\pi_i$, so $\pi_{i+1}=\arg\max M_i$ guarantees $\eta(\pi_{i+1})-\eta(\pi_i)\ge M_i(\pi_{i+1})-M_i(\pi_i)\ge 0$. Monotonic improvement, for *any* stochastic policies.

The penalty is theoretically honest but practically useless: with $\gamma=0.99$, $(1-\gamma)^2=10^{-4}$, so $C\sim 10^4\varepsilon$ and the maximizer of $M_i$ barely moves — microscopic steps, training forever. The fix that makes TRPO usable is to stop treating the KL as a penalty with a fixed coefficient and treat it as a *constraint* with a budget I choose: solve $\max_\theta L_{\theta_{old}}(\theta)$ subject to $D_{KL}^{\max}(\theta_{old},\theta)\le\delta$. The radius $\delta$ is interpretable — it directly says how far the policy may move per update — and gives consistent step sizes across iterations and problems, where the penalty coefficient is hard to choose robustly and the theory's value is uselessly conservative. Two further reductions make it estimable: the per-state max KL becomes the *average* KL over the old policy's discounted state distribution, $\bar D_{KL}^\rho(\theta_{old},\theta)=\mathbb{E}_{s\sim d_{old}}[D_{KL}(\pi_{old}(\cdot|s)\|\pi_\theta(\cdot|s))]\le\delta$, a single scalar constraint; and the objective becomes an importance-weighted advantage over sampled data. Expanding $L$, absorbing the constant $1/(1-\gamma)$ (positive, irrelevant to the KL-normalized step), replacing $A_{old}$ by $Q_{old}$ (they differ by the action-independent $V_{old}(s)$), and rewriting the inner action sum as an importance-sampling expectation gives
$$\max_\theta\ \mathbb{E}_{s\sim d_{old},\,a\sim\pi_{old}}\!\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}\,A_{old}(s,a)\right] \quad\text{s.t.}\quad \mathbb{E}_{s\sim d_{old}}\!\left[D_{KL}(\pi_{old}(\cdot|s)\|\pi_\theta(\cdot|s))\right]\le\delta.$$

Solving this for a network every iteration is where the rest of the method lives. Because $\delta$ is small, the step stays near $\theta_{old}$ and local models suffice: linearize the surrogate, $L(\theta)\approx g^\top(\theta-\theta_{old})$ with $g=\nabla_\theta L|_{\theta_{old}}$; and quadraticize the constraint — $\bar D_{KL}$ is zero at $\theta_{old}$ *and* has zero gradient there (KL is minimized at the matching distribution), so the leading term is $\tfrac12(\theta-\theta_{old})^\top A(\theta-\theta_{old})$ with $A=\nabla^2_\theta\bar D_{KL}|_{\theta_{old}}$ — precisely the average Fisher information matrix. The subproblem $\max_x g^\top x$ s.t. $\tfrac12 x^\top A x\le\delta$ is a quadratic-constrained linear program; its Lagrangian optimum lies on the boundary with $g=\lambda A x$, so the direction is $x\propto A^{-1}g$ — the *natural gradient*, which falls out rather than being imposed. (A Euclidean trust region $\tfrac12\|x\|^2\le\delta$ would give back vanilla PG; dropping the constraint and fully maximizing $L$ gives policy iteration — three classical methods as limits of one update.) For the step length I do not leave a free learning rate: I saturate the constraint. With $x=A^{-1}g$ and $\theta=\theta_{old}+\beta x$, plugging into $\tfrac12\beta^2 x^\top A x=\delta$ gives
$$\beta = \sqrt{\frac{2\delta}{x^\top A x}},$$
so every step moves the policy a controlled KL distance regardless of task. This is exactly what fixed-step natural gradient lacks. Forming $A$ is a non-starter for tens of thousands of parameters, so I never materialize it: conjugate gradient solves $Ax=g$ using only Fisher-vector products $v\mapsto Av$, and I get those as a Hessian-vector product $Av=\nabla_\theta((\nabla_\theta\bar D_{KL})^\top v)$ — two backward passes, about one gradient in cost — built analytically (integrating over the action, which is lower-variance than the empirical score covariance) and estimated on a subsample of the batch since $A$ is only a metric, with a small damping $A\to A+\eta I$ to keep CG well-conditioned. The final safeguard closes the gap between the local models and the truth: the quadratic/linear models are only good near $\theta_{old}$, and $\beta$ was computed to sit exactly on the model boundary, so the true KL may exceed $\delta$ or the true surrogate may not improve. A backtracking line search tries $\theta_{old}+\alpha^j\beta x$ for $j=0,1,2,\dots$ and accepts the first step where the *true* surrogate improves and the *true* average KL is $\le\delta$, taking no step if none qualifies — verifying against reality and backing off where the model lied. The policy is a diagonal Gaussian with network mean $\mu_\theta(s)$ and state-independent learned log-standard-deviations for continuous control (closed-form KL, hence cheap $A$), or a softmax categorical for discrete actions, with advantages reduced by a learned value-function baseline fit by regression to returns.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    """State -> diagonal Gaussian over actions; state-independent log-std."""
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]; last = h
        layers += [nn.Linear(last, act_dim)]
        self.mu_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def dist(self, obs):
        return Normal(self.mu_net(obs), torch.exp(self.log_std))

    def logp(self, obs, act):
        return self.dist(obs).log_prob(act).sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]; last = h
        layers += [nn.Linear(last, 1)]
        self.v = nn.Sequential(*layers)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def set_flat_params(model, flat_params):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat_params[i:i + n].view_as(p)); i += n


def discount_cumsum(x, discount):
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


def gae(rewards, values, last_val, gamma=0.99, lam=0.97):
    """Generalized Advantage Estimation; returns advantages and value targets."""
    rews = np.append(rewards, last_val)
    vals = np.append(values, last_val)
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    adv = discount_cumsum(deltas, gamma * lam)
    ret = discount_cumsum(rews, gamma)[:-1]
    return adv, ret


def surrogate(policy, obs, act, adv, logp_old):
    ratio = torch.exp(policy.logp(obs, act) - logp_old)
    return (ratio * adv).mean()


def mean_kl(policy, obs, mu_old, std_old):
    d, d_old = policy.dist(obs), Normal(mu_old, std_old)
    return torch.distributions.kl_divergence(d_old, d).sum(-1).mean()


def fisher_vector_product(policy, obs, mu_old, std_old, v, damping=0.1):
    kl = mean_kl(policy, obs, mu_old, std_old)
    g = flat(torch.autograd.grad(kl, policy.parameters(), create_graph=True))
    gv = (g * v).sum()
    hv = flat(torch.autograd.grad(gv, policy.parameters(), retain_graph=True))
    return hv + damping * v


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone(); p = b.clone(); r_dot = torch.dot(r, r)
    for _ in range(iters):
        Ap = Avp(p)
        alpha = r_dot / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p; r -= alpha * Ap
        r_dot_new = torch.dot(r, r)
        if r_dot_new < tol:
            break
        p = r + (r_dot_new / r_dot) * p; r_dot = r_dot_new
    return x


def trpo_step(policy, obs, act, adv, logp_old, mu_old, std_old,
              delta=0.01, backtrack_coeff=0.8, backtrack_iters=10):
    L_old = surrogate(policy, obs, act, adv, logp_old)
    L_old_value = L_old.detach()
    g = flat(torch.autograd.grad(L_old, policy.parameters(), retain_graph=True))

    Avp = lambda v: fisher_vector_product(policy, obs, mu_old, std_old, v)
    x = conjugate_gradient(Avp, g)                       # x = A^{-1} g

    xAx = torch.dot(x, Avp(x))
    beta = torch.sqrt(2 * delta / (xAx + 1e-8))          # saturate KL budget
    full_step = beta * x

    old_params = flat([p.data for p in policy.parameters()])
    for j in range(backtrack_iters):                     # line search on true L, true KL
        set_flat_params(policy, old_params + (backtrack_coeff ** j) * full_step)
        with torch.no_grad():
            kl = mean_kl(policy, obs, mu_old, std_old)
            L_new = surrogate(policy, obs, act, adv, logp_old)
        if kl <= delta and L_new > L_old_value:
            return
    set_flat_params(policy, old_params)                  # reject: keep old policy


def fit_value(value_net, obs, returns, opt, iters=80):
    for _ in range(iters):
        opt.zero_grad()
        (((value_net(obs) - returns) ** 2).mean()).backward()
        opt.step()


def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    return out


def train(env, policy, value_net, epochs=50, steps=4000, gamma=0.99, lam=0.97):
    v_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        O, Ac, Rew, LogpOld, MuOld = [], [], [], [], []
        o, ep_rew, ep_val = reset_env(env), [], []
        for t in range(steps):
            ot = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                d = policy.dist(ot); a = d.sample()
                logp = d.log_prob(a).sum(-1); val = value_net(ot)
            o2, r, done, _ = step_env(env, a.numpy())
            O.append(o); Ac.append(a.numpy()); ep_rew.append(r); ep_val.append(val.item())
            LogpOld.append(logp.item()); MuOld.append(d.mean.numpy()); o = o2
            if done or t == steps - 1:
                with torch.no_grad():
                    last_val = 0.0 if done else value_net(torch.as_tensor(o, dtype=torch.float32)).item()
                adv, ret = gae(np.array(ep_rew), np.array(ep_val), last_val, gamma, lam)
                Rew.append((adv, ret)); o, ep_rew, ep_val = reset_env(env), [], []
        adv = np.concatenate([a for a, _ in Rew]); ret = np.concatenate([r for _, r in Rew])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(np.array(O), dtype=torch.float32)
        act = torch.as_tensor(np.array(Ac), dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        logp_old = torch.as_tensor(np.array(LogpOld), dtype=torch.float32)
        mu_old = torch.as_tensor(np.array(MuOld), dtype=torch.float32)
        std_old = torch.exp(policy.log_std).detach().expand_as(mu_old)

        trpo_step(policy, obs, act, adv_t, logp_old, mu_old, std_old, delta=0.01)
        fit_value(value_net, obs, ret_t, v_opt, iters=80)
```

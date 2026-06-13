The thing that keeps bothering me about the quantile agent is that I picked the number 200 by hand and then the network is stuck with it forever. Let me look hard at what that agent actually represents. It learns a uniform mixture of $N$ Diracs, $Z_\theta(x,a)=\frac1N\sum_{i=1}^N\delta_{\theta_i(x,a)}$, where each location $\theta_i$ is trained to sit at the midpoint quantile $\hat\tau_i=\frac{2i-1}{2N}$ of the return. The output layer is $|\mathcal A|\times N$, so the network emits exactly $N$ numbers per action, and those $N$ numbers are tied, one-to-one, to $N$ levels $\hat\tau_1,\dots,\hat\tau_N$ that I chose before training started. The approximation can never be finer than $N$, and to make it finer I have to widen the output layer and retrain from scratch. The resolution of the distribution is an *architectural* constant, not something that improves as the network gets bigger or trains longer. That's the wart. A 200-Dirac approximation of the return is a staircase with 200 steps no matter how much capacity the rest of the net has to spare.

So what is the object I'm really approximating? The locations are quantiles, and "the location of the $\tau$-quantile as a function of $\tau$" is just the quantile function $F_Z^{-1}(\tau)$ — the inverse CDF, a map from a probability $\tau\in[0,1]$ to a return value. The quantile agent samples this function at $N$ fixed grid points and learns those $N$ outputs. But $F_Z^{-1}$ is a *continuous* function of $\tau$. If I could learn the whole function — a map I can query at *any* $\tau$ — then I'm no longer pinned to a grid: I evaluate it at as many or as few points as I like, and the fidelity of the representation is set by how well the network approximates a curve, which is a matter of capacity and training, not of a hardcoded $N$. Let me chase that.

Write the target I want: $Z_\tau(x,a):=F_{Z(x,a)}^{-1}(\tau)$, so that for $\tau\sim U([0,1])$, the value $Z_\tau(x,a)$ is a genuine sample from the return distribution $Z(x,a)$. That last sentence is the whole reparameterization trick stated for returns: if I have the quantile function and I push a uniform random variable through it, I get a sample of the return. So instead of learning probabilities on fixed locations (categorical) or locations on fixed probabilities (quantile-on-a-grid), I learn a deterministic function that *reparameterizes* a base sample $\tau\sim U([0,1])$ into a return sample. The distribution is then defined *implicitly* — I never write down its density or its CDF, I only know how to sample it: draw $\tau$, evaluate the network. With enough capacity this can approximate any distribution over returns, because any distribution is the pushforward of a uniform through its own quantile function.

Now I have to make this a network and a loss, and there are two questions the grid agent answered by construction that I now have to answer by design: how does $\tau$ enter the network, and what trains the function so that $Z_\tau(x,a)$ actually lands on $F^{-1}_{Z(x,a)}(\tau)$ for *every* $\tau$, not just a chosen few.

Take the training first, because it's the one I'm sure about. The quantile-regression argument doesn't care that I sampled $\tau$ from a grid — it cares only that I'm trying to hit the $\tau$-quantile. For any single level $\tau$, the $\tau$-quantile of a distribution is the minimizer of the asymmetric loss $\mathbb{E}_{\hat Z}[\rho_\tau(\hat Z-\theta)]$ with $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, and its error-gradient $\tau-\mathbb{1}_{u<0}$ depends only on the sign of $u$, so one sample gives an unbiased gradient. The Bellman version: build the sampled TD error from *two independent* base samples $\tau$ and $\tau'$,
$$\delta^{\tau,\tau'}_t=r_t+\gamma\,Z_{\tau'}(x_{t+1},\pi(x_{t+1}))-Z_{\tau}(x_t,a_t),$$
where $Z_{\tau}(x_t,a_t)$ is my network's prediction at level $\tau$ and $Z_{\tau'}(x',\pi(x'))$ is a bootstrapped target-network sample at level $\tau'$. The prediction at level $\tau$ is regressed toward the bootstrapped target at level $\tau'$ using the quantile loss at level $\tau$. Crucially, $\tau$ and $\tau'$ are *drawn fresh*, from a continuous distribution, each update — not read off a fixed comb. So the function gets trained at a dense, ever-changing set of levels, which is exactly what forces it to be right *as a function* and not just at 200 pinned points.

The kink in $\rho_\tau$ at $u=0$ hurts a deep network — the gradient magnitude stays constant ($\tau$ or $1-\tau$) right down to zero error, so the locations jitter. Round it off the same way the grid agent did: Huberize with threshold $\kappa$, quadratic inside $[-\kappa,\kappa]$ and linear outside,
$$\mathcal L_\kappa(u)=\begin{cases}\tfrac12u^2,&|u|\le\kappa\\[2pt]\kappa(|u|-\tfrac12\kappa),&|u|>\kappa,\end{cases}\qquad
\rho^\kappa_\tau(u)=\big|\tau-\mathbb{1}\{u<0\}\big|\,\frac{\mathcal L_\kappa(u)}{\kappa},$$
with $\kappa=1$ the usual choice, recovering the gradient-clipped squared error inside the band but with the asymmetric quantile weight. The division by $\kappa$ keeps the large-error slope independent of the chosen transition width, so changing $\kappa$ changes where the quadratic part gives way to the linear part rather than changing the whole loss scale.

Here's where I have to decide the sample counts, because now they're *knobs* rather than the fixed $N$. I sample $N$ values of $\tau$ for the predictions and $N'$ values of $\tau'$ for the targets, and form the loss over all pairs:
$$\mathcal L=\frac1{N'}\sum_{i=1}^{N}\sum_{j=1}^{N'}\rho^\kappa_{\tau_i}\big(\delta_t^{\tau_i,\tau_j'}\big).$$
Sum over the predicted levels $i$ (each predicted quantile is regressed at its own level), average over the target samples $j$ (they're a Monte-Carlo estimate of the bootstrapped distribution, so I average, hence the $1/N'$). Think about what $N$ and $N'$ each control. $N$ is how many points of my own quantile function I touch per update — it's like how much of the curve I supervise each step, so larger $N$ spends compute on shaping more of the function at once. $N'$ is how many target samples I average to estimate the bootstrapped distribution — it's a variance-reduction count on the regression target, like a minibatch size for the target, so its marginal value should drop once the target estimate is quiet enough. Push $N=1$ in my head: then I touch a single random point of my quantile function per update, which is close to a non-distributional agent in the number of loss terms, so it is a clean diagnostic for whether the benefit is merely an auxiliary-loss effect from having many heads. For the working agent I want both counts moderate: enough $N$ to shape several parts of the curve, enough $N'$ to denoise the target, but nowhere near a fixed 200-location output layer. $N=N'=8$ is the balanced setting I land on.

Now the architecture: how does $\tau$ get into the network? The state side should stay as close as possible to the DQN machinery. I take $\psi(x)$ to be the fixed Nature-DQN encoder ending in the 512-dimensional ReLU feature vector; that is just choosing the boundary so the encoder includes the usual hidden layer, leaving the action head $f$ as a single linear map from 512 features to actions. I want to insert dependence on $\tau$ without rebuilding either, so I add a third function $\phi(\tau)$ that embeds the scalar level into the same 512-dimensional feature space, and combine it with $\psi(x)$ before the action head: $Z_\tau(x,a)\approx f\big(\psi(x)\,\square\,\phi(\tau)\big)_a$ for some combination $\square$. What should $\square$ be? Concatenation is the lazy choice, but with a single linear action head, if I merely concatenate $\psi(x)$ and $\phi(\tau)$ and hit them with one linear map, $\tau$ enters only additively: the output is (linear in $\psi$) + (linear in $\phi(\tau)$), and a $\tau$-dependent *shift* of every action's value can't reshape the quantile function per state — it can only slide the whole curve up and down. That's not enough; the *shape* of $F^{-1}$ has to change with $\tau$ in a state-dependent way. I need $\tau$ to *multiply* the state features so the interaction is there even through a shallow head. So take the combination to be the element-wise (Hadamard) product,
$$Z_\tau(x,a)\approx f\big(\psi(x)\odot\phi(\tau)\big)_a,$$
which lets $\phi(\tau)$ gate each feature of $\psi(x)$ — feature-wise multiplicative modulation, so even a single linear $f$ on top sees a genuinely $\tau$-conditioned input. A residual form $\psi\odot(1+\phi)$ or plain concatenation are conceivable, but the multiplicative one forces the interaction through the shallow $f$, which is the pressure point I need to solve.

What's a good embedding $\phi(\tau)$ for a scalar in $[0,1]$? Feeding the raw scalar $\tau$ through a linear layer is weak — one input dimension, so the embedding is rank-one in $\tau$ and can't represent a rich dependence. I want a *basis expansion* of $\tau$: lift the scalar into many features that vary at different rates, so a linear layer on top can synthesize an arbitrary smooth function of $\tau$. Cosines of increasing frequency are the natural basis on an interval — a Fourier-type feature map — and they're bounded, which keeps the embedding well-scaled. So expand $\tau$ into $n$ cosine features $\cos(\pi i\tau)$ for $i=1,\dots,n$; if I instead index from $0$, the constant cosine can be absorbed into the bias term, so the implemented convention is the same modeling choice. Then a linear layer with a ReLU maps those features into the feature dimension $d=512$ to match $\psi(x)$:
$$\phi_j(\tau)=\operatorname{ReLU}\!\Big(\sum_{i=1}^{n}\cos(\pi i\tau)\,w_{ij}+b_j\Big).$$
An embedding dimension $n=64$ cosines is plenty of frequency content to describe how the quantile function bends with $\tau$, and it's cheap. The linear-then-ReLU after the cosines lets the network pick which frequencies matter and recombine them, and the ReLU keeps the gating non-negative-ish in the right regime; mainly it gives $\phi$ the same nonlinearity budget as the rest of the head. So the per-$\tau$ cost is one tiny cosine expansion plus one linear layer, shared across all the $\tau$ samples in a batch.

Let me make sure I haven't quietly added capacity that defeats the point. The torso is untouched. The head is the same shallow $f$. The only new parameters are the cosine-embedding's linear layer ($n\to d$) and whatever maps the modulated features to actions — a small addition, and the representational gain comes from *learning a function of $\tau$*, not from a wider output layer. In fact, where the grid agent's output layer was $|\mathcal A|\times N$ (large, since $N$ was big), mine is $|\mathcal A|$ per evaluated $\tau$, reused across samples — so I've *removed* the $N$-fold output blowup and replaced it with a single small embedding branch. Good.

Now the policy. The grid agent acts on the mean of its $N$ locations. I can do the same by Monte-Carlo: the mean of $Z(x,a)$ is $\mathbb{E}_{\tau\sim U([0,1])}[Z_\tau(x,a)]$, so I approximate it with $K$ fresh samples and take the argmax,
$$\tilde\pi(x)=\argmax_a\frac1K\sum_{k=1}^K Z_{\tilde\tau_k}(x,a),\qquad \tilde\tau_k\sim U([0,1]),$$
with $K=32$ samples — enough to make the action choice a Monte-Carlo estimate of the mean rather than a single noisy draw. The bootstrapped target uses this greedy action: $\pi(x')=\argmax_a\frac1K\sum_k Z_{\tilde\tau_k}(x',a)$, plugged into the TD error above with $\gamma=0$ at terminals. Everything else — the replay buffer, $\epsilon$-greedy, the periodic target copy — stays as in DQN.

But wait — once I have the whole quantile function, I notice I've quietly gained something the mean throws away. The mean is $\int_0^1 F_Z^{-1}(u)\,du$, a *uniform* average over quantile levels. Nothing forces the policy to weight the levels uniformly. Suppose the policy samples quantile levels from some distribution $\mu$ on $[0,1]$ and acts on $\frac1K\sum_k Z_{u_k}(x,a)$ with $u_k\sim\mu$ instead of $u_k\sim U([0,1])$. In the limit this estimates
$$Q_\mu(x,a)=\int_0^1 F^{-1}_{Z(x,a)}(u)\,d\mu(u),$$
a distorted expectation, an integral of the quantile function against a non-uniform weighting over probability levels. If I prefer to implement this with a deterministic transform, I choose a monotone map $\beta$ and set $u=\beta(\tilde\tau)$ for $\tilde\tau\sim U([0,1])$; what matters is the distribution over the final quantile level $u$. Put more mass near low quantiles and the policy becomes risk-averse; put more mass near high quantiles and it becomes risk-seeking. CVaR at level $\eta$ is the clean lower-tail example: sample $u\sim U([0,\eta])$, giving $\frac1\eta\int_0^\eta F_Z^{-1}(u)\,du$ and therefore only the worst $\eta$-fraction of outcomes. The risk-neutral agent is just $\mu=U([0,1])$. For the value-maximizing agent I keep the uniform policy — the mean-greedy policy — so the comparison to the grid agent isolates the effect of the implicit, continuous representation; the risk knob is a bonus the representation unlocks.

Let me sanity-check the whole loop end to end against the failure I started from. The grid agent's fidelity was capped at a hand-set $N$ and its policy read only the mean. I now (1) learn $F_Z^{-1}$ as a function via a cosine-embedded $\tau$ multiplied into the state features, so fidelity is bounded by capacity and training, not by an output-layer width; (2) train it with the same unbiased quantile-Huber regression but on *freshly sampled* levels each step, $N$ predictions against $N'$ targets, all pairs; (3) act on a $K$-sample Monte-Carlo mean for risk-neutral, or on a reweighted sample for risk-sensitive. As a distributional generalization of DQN it adds only the $\phi(\tau)$ branch and the Hadamard product; everything else in the agent is unchanged, which is the property I wanted — a minimal, single-axis change over the grid agent that removes its one structural limitation.

One more check on the samples per update, because it's the design choice most likely to bite. If $N'$ were too small the target estimate $\frac1{N'}\sum_j(\cdot)$ would be noisy and the regression would chase noise; if $N$ were too small I'd shape too little of the curve per step and learn slowly. Setting both to $8$ trades a little compute for dense-enough supervision of the function and a quiet-enough target. The cost per update is $N\times N'$ pairwise terms, which at $8\times8=64$ is comparable to the grid agent's per-update work and far below its $200$-wide output.

Now the concrete agent on the fixed 512-dimensional DQN encoder is small enough to write down. The encoder $\psi$ returns a 512-vector, $\phi$ maps the 64 cosine features of each $\tau$ into that same 512-vector space, the Hadamard product creates the state-dependent quantile features, and the remaining action head is a single linear map:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 8
N_PRIME = 8
K = 32
N_COS = 64
KAPPA = 1.0


class FixedDQNEncoder(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


class QuantileReturnNet(nn.Module):
    def __init__(self, n_actions, d=512):
        super().__init__()
        self.encoder = FixedDQNEncoder(d)
        self.embed = nn.Linear(N_COS, d)
        self.head = nn.Linear(d, n_actions)
        self.register_buffer("freqs", torch.arange(1, N_COS + 1, dtype=torch.float32) * torch.pi)

    def z(self, x, taus):
        psi = self.encoder(x).unsqueeze(1)
        cos = torch.cos(taus.unsqueeze(-1) * self.freqs)
        phi = F.relu(self.embed(cos))
        return self.head(psi * phi)

    def greedy_action(self, x):
        taus = torch.rand(x.shape[0], K, device=x.device)
        return self.z(x, taus).mean(dim=1).argmax(dim=1)


def quantile_huber(td, taus, kappa=KAPPA):
    huber = torch.where(
        td.abs() <= kappa,
        0.5 * td.pow(2),
        kappa * (td.abs() - 0.5 * kappa),
    )
    weight = (taus.unsqueeze(-1) - (td.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean(dim=2).sum(dim=1).mean()


def loss(online, target, obs, actions, rewards, next_obs, dones, gamma):
    b = obs.shape[0]
    with torch.no_grad():
        next_actions = target.greedy_action(next_obs)
        tau_prime = torch.rand(b, N_PRIME, device=obs.device)
        z_next = target.z(next_obs, tau_prime)
        z_next = z_next.gather(2, next_actions.view(b, 1, 1).expand(b, N_PRIME, 1)).squeeze(2)
        target_z = rewards.view(b, 1) + gamma * (1.0 - dones.float()).view(b, 1) * z_next
    tau = torch.rand(b, N, device=obs.device)
    z = online.z(obs, tau)
    pred = z.gather(2, actions.view(b, 1, 1).expand(b, N, 1)).squeeze(2)
    delta = target_z.unsqueeze(1) - pred.unsqueeze(2)
    return quantile_huber(delta, tau)
```

The causal chain in one breath: the grid quantile agent caps its return-distribution resolution at a hand-chosen $N$ and reads only the mean, so I learn the *quantile function itself* — a map from a base sample $\tau\sim U([0,1])$ to a return, which defines the distribution implicitly and whose fidelity is bounded by capacity, not $N$; I inject $\tau$ via a cosine embedding multiplied (Hadamard) into the state features so a shallow head still sees a state-dependent $\tau$-interaction; I train it with the same unbiased quantile-Huber loss but on freshly sampled levels each update, summing over $N$ prediction samples and averaging over $N'$ target samples; I act on a $K$-sample Monte-Carlo estimate of the mean for the risk-neutral agent — and because a non-uniform reweighting of the $\tau$-sampling turns that mean into a distorted expectation, the very same network yields an entire family of risk-sensitive policies by changing only how $\tau$ is sampled.

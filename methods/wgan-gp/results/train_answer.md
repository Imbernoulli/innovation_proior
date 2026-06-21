The thing that kept defeating me was not getting a GAN to work once, but getting it to work twice: a careful architecture with the right batch normalization and learning rate trains, and then I drop the normalization, swap a nonlinearity, or make the critic deeper, and it collapses or stalls. So I treated the instability as the central object of study rather than a nuisance. The original game has the generator minimize, against an optimal discriminator, the Jensen-Shannon divergence between the data distribution $P_r$ and the model distribution $P_g$. The trouble is geometric: $P_r$ is natural images, a thin low-dimensional sheet in a huge pixel space, and $P_g$ is a low-dimensional noise pushed through $G$, also a thin sheet; two thin sheets in a high-dimensional room generically do not intersect, or intersect only on a measure-zero set. Where the supports are disjoint a high-capacity discriminator separates them perfectly, JS is pinned at its maximum $\log 2$ and locally constant, and so its gradient with respect to the generator's parameters is zero almost everywhere — the better the discriminator, the flatter the loss the generator sees. The non-saturating $-\log D$ trick rescales that gradient but cannot change the geometry, and KL-style objectives are no escape: against singular supports the KL is infinite or undefined rather than a finite, distance-aware slope. The disease is the divergence itself, because it only registers overlap.

What I need is a distance between distributions that decreases smoothly as one sheet slides toward the other even while they remain disjoint. The Earth-Mover (Wasserstein-1) distance is exactly that — the minimum total work, mass times distance carried, to reshape $P_g$ into $P_r$, written as $W(P_r,P_g)=\inf_{\gamma\in\Pi(P_r,P_g)}\mathbb{E}_{(x,y)\sim\gamma}\,[\,\lVert x-y\rVert\,]$ over couplings $\gamma$ with marginals $P_r,P_g$. Translate $P_g$ by a small $\varepsilon$ toward $P_r$ and the optimal plan's cost drops by about $\varepsilon$, so $W$ behaves like the literal distance between the sheets and stays differentiable almost everywhere as $G$ moves. The infimum over couplings is intractable, but Wasserstein-1 has the Kantorovich-Rubinstein dual $W(P_r,P_g)=\sup_{\lVert f\rVert_L\le 1}\mathbb{E}_{x\sim P_r}[f(x)]-\mathbb{E}_{\tilde x\sim P_g}[f(\tilde x)]$, a maximization over $1$-Lipschitz functions $f$ — and a neural network is a function class. So I parameterize $f$ as a net, the *critic* (no sigmoid, an unbounded real score, not a classifier), and play $\min_G\max_{D\in\{1\text{-Lipschitz}\}}\mathbb{E}_{x\sim P_r}[D(x)]-\mathbb{E}_{\tilde x\sim P_g}[D(\tilde x)]$, where the inner max estimates $W$ and the outer min drags $P_g$ along its smooth slope. Everything then hinges on the one clause I keep waving at: how do I keep a network $1$-Lipschitz during training? The original Wasserstein critic clamps every weight into a box $[-c,c]$ after each step, which keeps $f$ in *some* $k$-Lipschitz class. It trains, but watching it closely it fails in two diagnosable ways. Freeze the generator at real-data-plus-noise so I know the target, train a clipped critic to optimality on a toy 2D distribution, and the critic comes back embarrassingly simple — it gets the gross location and ignores higher moments; the weight histogram piles up at $\pm c$ because to maximize the score gap the net slams every weight against the box walls and spends all its capacity on being maximally steep, none on shape. And on a deep critic with no batch norm, the gradient of the critic loss through depth either shrinks or grows geometrically; trying $c\in\{10^{-1},10^{-2},10^{-3}\}$ each gives an exponential, just a different base, because $c$ is simultaneously controlling the Lipschitz constant and the conditioning of backprop, with no value that keeps both happy. Both failures share a root: I am constraining a property of the *function* by bludgeoning the *weights*, and the weights are a terrible handle on it.

So I propose WGAN-GP — Wasserstein GAN with gradient penalty — which enforces the actual constraint directly and softly. The local form of $1$-Lipschitzness for a differentiable critic is $\lVert\nabla_x D(x)\rVert\le 1$ everywhere, so rather than clip weights I penalize the critic's input-gradient norm. The question is what value to push the norm toward and where to sample, and the answer comes from looking at what the dual optimum $f^\*$ actually looks like. Let $\pi$ be the optimal coupling, with $\tilde x$ the generated endpoint and $x$ the real endpoint; dual tightness gives $f^\*(x)-f^\*(\tilde x)=\lVert x-\tilde x\rVert$ for $\pi$-almost-every moved pair — the function gains exactly the transport cost over a coupled pair. Parameterize the segment $x_t=(1-t)\tilde x+t\,x$ and set $\psi(t)=f^\*(x_t)-f^\*(\tilde x)$, so $\psi(0)=0$ and $\psi(1)=\lVert x-\tilde x\rVert$. Lipschitzness makes $\psi$ itself $\lVert x-\tilde x\rVert$-Lipschitz in $t$, and splitting the rise, $\lVert x-\tilde x\rVert=(\psi(1)-\psi(t))+(\psi(t)-\psi(0))\le(1-t)\lVert x-\tilde x\rVert+t\lVert x-\tilde x\rVert=\lVert x-\tilde x\rVert$; both upper bounds already sum to the full endpoint rise, so neither can be slack, forcing $\psi(t)=t\lVert x-\tilde x\rVert$. Thus $f^\*$ rises *exactly linearly* along the segment with no slack to wander. Turning that into a gradient statement, the unit direction toward the real endpoint is $v=(x-x_t)/\lVert x-x_t\rVert=(x-\tilde x)/\lVert x-\tilde x\rVert$ at every interior $t$, and moving a small distance $h$ along $v$ advances $t$ by $h/\lVert x-\tilde x\rVert$, so $f^\*(x_t+hv)-f^\*(x_t)=h$ and the directional derivative is exactly $1$. Since $f^\*$ is $1$-Lipschitz, $\lVert\nabla f^\*(x_t)\rVert\le 1$; decomposing by Pythagoras, $\lVert\nabla f^\*(x_t)\rVert^2=\langle v,\nabla f^\*\rangle^2+\lVert\nabla f^\*-\langle v,\nabla f^\*\rangle v\rVert^2=1+\lVert\nabla f^\*-v\rVert^2$, and since the left side is $\le 1$ the orthogonal remainder must vanish, giving

$$\nabla f^\*(x_t)=v=\frac{x-x_t}{\lVert x-x_t\rVert},\qquad \lVert\nabla f^\*(x_t)\rVert=1.$$

So the optimal critic does not merely *satisfy* $\lVert\nabla f\rVert\le 1$; on the transport segments its gradient has norm *exactly* $1$ and points from the generated endpoint toward the real one. That answers both questions at once. The penalty target should be norm equal to $1$ — pinning it there is not an arbitrary over-constraint, since the true optimum already has unit-norm gradients on those segments — and the place to enforce it is along straight lines between generated and real points. I cannot sample the unknown optimal coupling, so the practical surrogate is to draw $x\sim P_r$, $\tilde x\sim P_g$, $\varepsilon\sim U[0,1]$, and form $\hat x=\varepsilon x+(1-\varepsilon)\tilde x$, spraying points along real/fake segments — the in-between region where the critic's geometry controls the generator's gradient. The critic loss to minimize is then

$$L=\mathbb{E}_{\tilde x\sim P_g}[D(\tilde x)]-\mathbb{E}_{x\sim P_r}[D(x)]+\lambda\,\mathbb{E}_{\hat x\sim P_{\hat x}}\big[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2-1)^2\big],$$

with the generator minimizing $-\mathbb{E}_z[D(G(z))]$.

A few choices in this objective are load-bearing. The penalty is *two-sided*, $(\lVert\nabla D\rVert-1)^2$, rather than the one-sided hinge $(\max(0,\lVert\nabla D\rVert-1))^2$ that punishes only slopes above $1$: the one-sided form is the literal relaxation of $\lVert\nabla D\rVert\le 1$, but the transport geometry just showed the optimum wants slope exactly $1$ on the relevant segments, and leaving below-one slopes unpenalized permits a critic too flat precisely where the generator needs a direction. The coefficient $\lambda$ is the single knob — too small and the critic drifts off $1$-Lipschitz so the loss stops being a Wasserstein distance, too large and the penalty starves the critic's expressiveness — and $\lambda=10$ is a fixed default that corrects slope violations while leaving the score difference visible, replacing the clip threshold's knife-edge with a stable constant. Critically, the penalty is a *per-example* statement, $\nabla_{\hat x}D(\hat x)$ for each single input, but batch normalization makes one example's output depend on the whole batch and leaks the gradient through batch statistics, so $\nabla_{\hat x}D(\hat x)$ is no longer well defined per example; I therefore drop batch norm from the critic and use layer normalization, which normalizes each example over its own features and keeps the penalty valid. There is also a smoothness subtlety: the parameter-gradient of $L$ contains $\nabla_w(\lVert\nabla_{\hat x}D(\hat x)\rVert^2)$, which involves *second* derivatives of $D$ and hence of the activations; ReLU's second derivative is zero almost everywhere so the autodiff path is usable at almost every sampled point, but a genuinely non-smooth activation can bite, and the clean fix is a smooth analogue such as $\mathrm{softplus}(2x+2)/2-1$. Finally, the outer min over $G$ only minimizes a true Wasserstein distance if the inner max over $D$ has actually been solved, so I push the critic closer to optimality with $n_{\text{critic}}=5$ steps per generator step — something clipping prevented because of its gradient pathologies and the penalty now allows — and I use Adam with the first moment killed, $\beta_1=0$, $\beta_2=0.9$, learning rate $10^{-4}$, because the adversarial target is non-stationary and first-moment momentum overshoots a moving target.

```python
import torch
import torch.nn as nn
import torch.autograd as autograd


def lipschitz_enforcement(critic, real, fake, lambda_gp=10.0):
    batch_size = real.size(0)
    # one eps per example, broadcast over feature dims, uniform on [0, 1]
    eps = torch.rand(batch_size, *([1] * (real.dim() - 1)), device=real.device)
    x_hat = eps * real + (1 - eps) * fake          # point on real<->fake segment
    x_hat.requires_grad_(True)

    d_hat = critic(x_hat)
    grads = autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,      # second-order: penalty differentiable in params
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)               # ||grad_{x_hat} D(x_hat)||_2
    return lambda_gp * ((grad_norm - 1) ** 2).mean()   # two-sided penalty


def train_step(generator, critic, opt_g, opt_c, real, noise_dim,
               n_critic=5, lambda_gp=10.0):
    device = real.device
    for _ in range(n_critic):                      # train critic to near-optimum
        z = torch.randn(real.size(0), noise_dim, device=device)
        fake = generator(z).detach()
        loss_c = critic(fake).mean() - critic(real).mean()   # Wasserstein loss
        loss_c = loss_c + lipschitz_enforcement(critic, real, fake, lambda_gp)
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        # no weight clipping: the penalty replaces it

    z = torch.randn(real.size(0), noise_dim, device=device)
    loss_g = -critic(generator(z)).mean()          # generator step
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return loss_c.item(), loss_g.item()


# Critic: scalar output, NO sigmoid, NO batch norm (use nn.LayerNorm / per-example
# normalization). Momentum-free Adam for the non-stationary adversarial objective:
#   opt_c = torch.optim.Adam(critic.parameters(),    lr=1e-4, betas=(0.0, 0.9))
#   opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
```

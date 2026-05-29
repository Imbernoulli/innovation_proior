OK, let me think this through from scratch. I want to learn to generate images. The honest version of the problem is: there's a real data distribution P_r over pixel space, and I have a generator — a neural net g_theta that takes noise z ~ p(z) and outputs a sample, defining a model distribution P_theta as the law of g_theta(Z). I want P_theta close to P_r. The whole game is "close" — what distance do I put between two distributions and descend on?

The textbook reflex is maximum likelihood: pick a parametric density P_theta, maximize the average log P_theta(x) over data, which asymptotically minimizes KL(P_r || P_theta). Fine. But stare at the geometry for a second. Natural images don't fill up the 64×64×3 cube; they sit on some thin, curved, low-dimensional sheet inside it. And my generator? It maps a low-dimensional z through a smooth net, so g_theta(Z) also concentrates on a low-dimensional sheet — at most as many dimensions as z has. So I have two thin sheets floating in a huge ambient space. Generic position: two low-dimensional sheets in a high-dimensional space either miss each other or cross on something even thinner — a set of measure zero. Either way, the model puts no mass where the data is and vice versa. There's no density of P_theta with respect to P_r — the Radon–Nikodym derivative doesn't exist — so KL(P_r||P_theta) is +∞, and the log-likelihood is −∞. Maximum likelihood is literally undefined for the object I actually have.

The standard escape is to smear the model with Gaussian noise so it has a density everywhere. But how much noise? People have measured this — to make the likelihood behave you need something like σ ≈ 0.1 per pixel on images normalized to [0,1]. That's enormous; it visibly blurs every sample, so much that when people report samples they quietly turn the noise off and only keep it for the likelihood number. That tells me the noise isn't a modeling choice, it's a crutch to rescue a distance that was the wrong distance to begin with. I don't want to fix the model to suit the distance. I want to fix the distance.

So drop densities. The generator already gives me an implicit distribution I can sample from; that's the right primitive for manifold-supported data. Now the real question, the only question, is: which notion of distance rho(P_theta, P_r) do I minimize? The choice of distance isn't cosmetic, it decides whether the loss even has a gradient. A distance defines which sequences of distributions count as converging: P_t -> P_inf means rho(P_t, P_inf) -> 0. A *weaker* distance calls more sequences convergent. And I want theta -> P_theta to be a continuous map so that theta -> rho(P_theta, P_r) is a continuous loss — because if it's continuous (and a.e. differentiable) I can do gradient descent on it, and if it's not, I can't. The weaker the distance, the easier it is for P_{theta_t} to converge when theta_t converges, so the easier it is for that loss to be continuous. So among the standard choices I should be hunting for the weakest useful distance, not the strongest one.

Let me get concrete about why the usual distances fail, because I want to feel the failure, not just assert it. Take the simplest possible disjoint-support toy: Z ~ U[0,1], let P_0 be the law of (0, Z) — a unit segment on the y-axis — and P_theta the law of (theta, Z) — the same segment shifted to x = theta. As theta -> 0 the two segments slide on top of each other; intuitively the distributions get "closer." What do the distances say?

Total variation: δ(P_0,P_theta) = sup_A |P_0(A) − P_theta(A)|. For any theta ≠ 0 the two segments are disjoint, so take A = the right segment: P_theta(A)=1, P_0(A)=0, δ=1. At theta=0 they coincide, δ=0. So δ jumps from 1 to 0. Discontinuous.

KL: for theta ≠ 0 there are points with P_theta>0 but P_0=0, so KL(P_theta||P_0)=+∞, and symmetrically the other way. At 0 it's 0. So KL is +∞ then 0. Useless.

JS: JS = ½KL(P_0||P_m)+½KL(P_theta||P_m) with P_m the mixture. For theta≠0 the supports are disjoint, each piece is fully on its own half of the mixture, and the arithmetic gives JS = log 2 for every theta ≠ 0, and 0 at theta=0. So JS is the constant log 2 on a punctured neighborhood of 0 and drops to 0 only exactly at 0. Flat everywhere ⇒ zero gradient everywhere ⇒ gradient descent has literally nothing to follow, and there's a discontinuity at the answer.

Now the Earth-Mover / Wasserstein-1 distance. Definition: W(P_r,P_g) = inf over couplings γ of E_{(x,y)~γ}[||x−y||], where a coupling is a joint distribution γ(x,y) whose marginals are P_r and P_g. Read it as the cheapest plan to physically haul the mass of one pile into the shape of the other, cost = mass × distance moved. For the two segments, the obvious plan moves each point (0, z) straight across to (theta, z): every unit of mass travels |theta|, so W(P_0,P_theta) = |theta|. And you can't do better, so that's the inf. So W = |theta|: continuous, and its derivative is sign(theta) — a clean, constant-magnitude gradient pointing me toward theta=0 from either side. This is exactly the case where every other distance died, and W sails through. As theta_t -> 0, P_{theta_t} converges to P_0 under W and under nothing else here.

That's the whole thesis in one example, and it's not just an artifact of disjointness — the same thing happens whenever the two manifolds intersect only on a measure-zero set, which is the generic case for image manifolds. So W is the distance I want. The question now is whether this niceness is real in general or an artifact of a 1-D toy.

Let me try to prove W gives a continuous loss for an actual generator. I want: if g_theta is continuous in theta, then theta -> W(P_r, P_theta) is continuous. The useful thing about W is that I get to *choose* a coupling, and any coupling gives an upper bound on the inf. So to bound W(P_theta, P_theta') I don't optimize over couplings — I just exhibit one. Use the same noise for both: let γ be the law of the pair (g_theta(Z), g_{theta'}(Z)). Its marginals are exactly P_theta and P_theta', so it's a valid coupling, and

  W(P_theta, P_theta') ≤ E_{(x,y)~γ}[||x−y||] = E_z[ ||g_theta(z) − g_{theta'}(z)|| ].

Now if g is continuous in theta, then for each z, g_theta(z) -> g_{theta'}(z) as theta -> theta', so the integrand goes to 0 pointwise. The space X is compact, so distances are uniformly bounded by some M, and bounded convergence lets me pass the limit inside the expectation: E_z[||g_theta − g_{theta'}||] -> 0. Hence W(P_theta, P_theta') -> 0. And to get continuity of the actual loss I use the reverse triangle inequality, |W(P_r,P_theta) − W(P_r,P_theta')| ≤ W(P_theta, P_theta') -> 0. So W(P_r, P_theta) is continuous in theta. Good — and notice JS/KL fail this exact statement, with the parallel-lines example as the counterexample.

Can I get differentiability too? Strengthen "continuous" to "locally Lipschitz": suppose for each (theta, z) there's a local Lipschitz constant L(theta, z) with ||g_theta(z) − g_{theta'}(z)|| ≤ L(theta,z)·||theta − theta'|| locally, and crucially that E_z[L(theta,z)] < ∞ — call it L(theta). Then the same coupling bound gives E_z[||g_theta − g_{theta'}||] ≤ ||theta−theta'||·L(theta), so |W(P_r,P_theta) − W(P_r,P_theta')| ≤ L(theta)·||theta − theta'||: W is locally Lipschitz in theta. A locally Lipschitz function is continuous everywhere and, by Rademacher's theorem, differentiable almost everywhere. That's exactly the regularity I need for gradient descent — a gradient exists at almost every theta.

And does a real neural net satisfy E_z[L(theta,z)] < ∞? Take g_theta with smooth Lipschitz nonlinearities. Its Jacobian in (theta, z) is a product/sum of weight matrices and the diagonal Jacobians of the nonlinearities; if the nonlinearity is L-Lipschitz then each diagonal block has norm ≤ L, and writing f_{1:k} for the first k layers, ||f_{1:k}(z)|| grows like ||z|| times a product of weight-matrix norms. Bounding the full gradient norm term by term gives ||∇_{theta,z} g_theta(z)|| ≤ C_1(theta) + C_2(theta)·||z||, where C_1, C_2 are products/sums of the weight-matrix norms times powers of L. So if the prior has E[||z||] < ∞ — Gaussian, uniform, anything sane — then E_z[L(theta,z)] ≤ C_1(theta) + C_2(theta)·E[||z||] < ∞. The assumption holds for any reasonable feedforward net. So W(P_r,P_theta) really is a continuous, almost-everywhere-differentiable loss for the generators I actually use. None of this is true for JS.

Let me also pin down the hierarchy so I know what I'm trading. Convergence in KL implies convergence in JS/TV, which implies convergence in W, and the implications are strict — W is the weakest among these. Write total variation as δ(P,Q)=sup_A |P(A)-Q(A)|. Pinsker gives δ(P,Q) <= sqrt(KL(P||Q)/2). For JS, with M=(P+Q)/2, I have δ(P,M)=δ(P,Q)/2 and δ(Q,M)=δ(P,Q)/2; applying Pinsker to both mixture KLs gives δ(P,Q) <= sqrt(2 JS(P,Q)), so JS -> 0 forces δ -> 0, and if δ -> 0 then both distributions are close to their mixture, so JS -> 0 as well. On a compact space with diameter B, a maximal coupling leaves only δ mass unmatched, so W(P,Q) <= B·δ(P,Q). So the order is KL ⇒ {JS, TV} ⇒ W, strictly. W -> 0 is equivalent to convergence in distribution, and a sequence can converge weakly while staying at TV-distance 1 (the parallel lines again). W converges in strictly more situations. That's precisely why it's the one that keeps giving me a usable loss when the supports don't overlap, and why "switch to a different f-divergence" can't save me — JS, reverse-KL, forward-KL, total variation, every density-ratio functional lives at the strong end and shares the flat-loss disease. The fix has to change the *geometry* of the comparison, not just the divergence label, and W does because it measures transport cost — it uses the metric on X — instead of a pointwise density ratio.

So I've decided: minimize W(P_r, P_theta). Now the trouble. The definition is an infimum over all couplings γ with the right marginals. That's a search over an enormous space of joint distributions — completely intractable; I can't enumerate transport plans for image distributions. The primal form is a dead end for computation. I need another handle on the same number.

Wasserstein-1 is an optimal-transport problem, and optimal transport is a linear program — minimize a linear cost over the coupling polytope. Linear programs have duals. The dual of this particular transport LP is the Kantorovich–Rubinstein duality (a classical result, in Villani): for the cost ||x−y||,

  W(P_r, P_g) = sup over 1-Lipschitz f of  E_{x~P_r}[f(x)] − E_{x~P_g}[f(x)],

the sup taken over all functions f: X -> R with ||f||_L ≤ 1, i.e. |f(x)−f(y)| ≤ ||x−y||. Let me make sure I believe the shape of this, because it's the load-bearing step. In the primal I pay ||x−y|| for every unit moved from x to y, and I must respect the marginals. Dualize the marginal constraints with a potential function f: the constraint that γ moves mass at cost ||x−y|| turns, in the dual, into the requirement that the potential never claims a price difference bigger than the transport cost — f(x) − f(y) ≤ ||x−y|| for all x,y, which is exactly 1-Lipschitz. And the dual objective collects the potential against the two marginals: E_{P_r}[f] − E_{P_g}[f]. So the impossible inf over couplings becomes a sup over a single scalar function constrained to be 1-Lipschitz. That's a massive simplification: instead of a joint distribution I'm optimizing one function. (And note if I relax to K-Lipschitz functions I just get K·W, a harmless constant rescaling.)

This is a beautiful reframing because of what the sup-form *is*. It's an integral probability metric: pick a class of functions F and define d_F(P,Q) = sup_{f∈F} E_P[f] − E_Q[f]. W is the special case F = 1-Lipschitz functions. Different F give wildly different distances — F = functions bounded in [−1,1] gives twice total variation, the same strong topology up to a constant; the unit ball of an RKHS gives MMD, which is closed-form via a kernel but costs O(samples²) and needs huge batches in high dimensions to even be a reliable statistic. The Lipschitz ball is the sweet spot: it's the class whose IPM is W, the weak distance. So the function class is the whole story, and I want the Lipschitz one.

Now, how do I actually take the sup over all 1-Lipschitz functions? I can't enumerate those either. The GAN-like thing I can reuse is function approximation: parameterize f by a neural network f_w and let the network do the maximization by gradient ascent. So I'd solve

  max_w  E_{x~P_r}[f_w(x)] − E_{z~p(z)}[f_w(g_theta(z))],

If the class were all K-Lipschitz functions, the supremum would be exactly K·W(P_r, P_theta). With a neural family inside that class, the maximized value is a lower estimate of K·W, and it gets better as the family and the inner optimization get better. This f_w looks like a GAN discriminator, but it is not a classifier. There's no sigmoid, no probability, no log; it outputs a raw real number, and the thing I read off is the *difference of its means* on real vs. fake. It's playing the role of the dual potential. I'll call it a critic, not a discriminator — it scores rather than classifies.

Two things I need to check before I trust this. First, can I differentiate W through this construction to get a gradient for the generator? Suppose for a fixed theta the inner max is attained by some critic f. Define V(f, theta) = E_{P_r}[f] − E_z[f(g_theta(z))]. The set of optimal critics is nonempty (Kantorovich–Rubinstein guarantees a maximizer on the compact X), and by the envelope theorem — when you differentiate the value of a maximization with respect to a parameter that only enters the objective, you can hold the maximizer fixed and differentiate the objective directly — I get ∇_theta W(P_r,P_theta) = ∇_theta V(f, theta) for the optimal f. The first term E_{P_r}[f] doesn't depend on theta, so

  ∇_theta W(P_r, P_theta) = −∇_theta E_z[f(g_theta(z))] = −E_z[ ∇_theta f(g_theta(z)) ],

where `∇_theta f(g_theta(z))` means the chain-rule derivative through the generator, `J_theta g_theta(z)^T ∇_x f(g_theta(z))` where the Lipschitz critic is differentiable. Pushing the gradient inside the expectation is justified because f∘g is Lipschitz hence (Rademacher) differentiable a.e.; the difference quotient is dominated locally by an integrable Lipschitz envelope, so dominated convergence applies, and Fubini lets me choose theta where the z-section of nondifferentiability has measure zero. So the generator gradient is just: backprop the critic's output through the generator and negate. Concretely, train the critic, then update theta to *raise* the critic's score on the fakes, which minimizes `-E_z[f(g_theta(z))]` and follows the descent direction for W.

With the JS/log-D objective, the better the discriminator gets, the closer the loss sits at the flat log 2 and the more the generator gradient vanishes — so you're forced to keep the discriminator deliberately weak and balance it against the generator, a miserable tightrope. With W it's the opposite: W is continuous and differentiable a.e., so the more I optimize the critic toward the true sup, the more accurate the W estimate and the better the gradient I hand the generator. There's no saturation to fear. So I *should* train the critic to (near-)optimality before each generator step — say several critic updates per generator update — instead of one cautious step. The thing that breaks GANs becomes the thing I lean into. As a bonus, since the critic's optimum is a genuine estimate of W (up to the constant K), the loss value itself finally means something: it tracks the distance, so a falling loss curve should track improving samples — a debugging signal GANs never had.

Second, and this is the real snag: the dual sup is over 1-Lipschitz f. If I just let f_w be an unconstrained net, the sup is +∞ (scale f up without bound) and I'm not computing W at all — I'm computing garbage. I must constrain f_w to be K-Lipschitz for some fixed K, so that the family is the Lipschitz ball (up to the irrelevant scale K) and the sup is finite. How do I keep a neural net Lipschitz?

Let me think about what controls a net's Lipschitz constant. It's roughly the product of the operator norms of the weight matrices times the Lipschitz constants of the nonlinearities. If I could bound every weight, the whole function's Lipschitz constant is bounded by a constant that depends only on the box I put the weights in, not on the particular weights — so all f_w in that box are K-Lipschitz for one common K. The crudest way to bound the weights: after each gradient update, clamp every weight back into a small box, w <- clip(w, −c, c), say c = 0.01. The parameter space W = [−c,c]^l is compact, every f_w is K-Lipschitz for a K determined by W, and I'm optimizing over the Lipschitz ball up to scale. So the recipe is: ascend the critic objective, then clip.

I want to be honest that clipping is a blunt instrument, because feeling its failure modes tells me how to set c. If c is too large, weights take forever to walk to the boundary of the box, so the effective constraint is loose and it's hard to drive the critic to optimality — and I just argued optimality is the whole point. If c is too small, then through many layers the products of small weights shrink the signal and the gradients vanish — especially without batchnorm or in recurrent nets. So there's a narrow band for c, and it interacts with depth and normalization. I tried fancier things — projecting weights onto a sphere — and they made little difference, so I'll keep clipping for its sheer simplicity, while treating clean Lipschitz enforcement as an unresolved engineering problem.

Let me sanity-check the constraint does the right thing with a clean picture: two well-separated Gaussians, train a GAN discriminator and a clipped critic each to optimality. The discriminator snaps to near 0/1 and saturates — flat with zero gradient between the modes, exactly the vanishing-gradient story. The clipped critic *can't* blow up — the Lipschitz cap forces it to grow at most linearly, so it settles into a smooth, roughly linear ramp between the two distributions, giving a clean nonzero gradient everywhere in between. That linear shape is the constraint biting in the most useful way. Good — the constraint isn't just making the sup finite, it's manufacturing exactly the gentle slope that the generator can descend.

One more property falls out. Mode collapse in a GAN comes from this: for a *fixed* discriminator, the generator's best response is to dump all its mass onto the single point the discriminator scores highest — a sum of deltas at the argmax. If the discriminator is held fixed while the generator chases it, the generator collapses. But here I train the critic toward optimality at every step, so there's no stale fixed target to chase for long; the critic re-optimizes against whatever the generator does. This attacks the fixed-discriminator collapse mechanism instead of relying on a deliberately undertrained discriminator.

The optimizer choice has the same flavor. The critic objective is highly non-stationary — its landscape shifts as the generator moves and as the weights get clipped. With a momentum optimizer like Adam (β1 > 0) on the critic, training would blow up; when it did, the cosine between the Adam step and the raw gradient turned negative — momentum was dragging the update against the gradient on this nonstationary loss. So momentum is the wrong tool here. RMSProp — which rescales by a running average of gradient magnitude but carries no momentum — handles nonstationary objectives well, so I'll use RMSProp with a small learning rate (5e-5). Plain, no momentum.

So let me assemble the whole loop. I keep the DCGAN convolutional body for both nets — the contribution is the objective and the training discipline, not a new architecture, and holding the body fixed isolates the effect of the distance. The generator is the usual transpose-conv stack mapping a 100-D Gaussian z to a 64×64×3 image. The critic is the DCGAN discriminator body with the final sigmoid removed and the output averaged to a single scalar — it's f_w. Per outer iteration: do n_critic critic steps, with 100 steps for the first 25 generator iterations and every 500th iteration when I really want the critic near-optimal before the generator moves; each critic step samples a real batch and a fake batch, descends `D(fake) - D(real)` by RMSProp, which is the same as ascending `E[f_w(real)] - E[f_w(fake)]`, and projects the weights into [−c,c]. Then one generator step descends `-D(fake)`, i.e. raises the critic's score on its own fakes, which by the gradient identity is a descent step on W.

At this point the loop is short enough to write down directly.

```python
import torch
import torch.nn as nn
import torch.optim as optim

nz, ngf, ndf, nc = 100, 64, 64, 3          # latent dim, gen/critic widths, image channels
c = 0.01                                    # weight-clip box: keeps the critic K-Lipschitz
n_critic = 5                                # train the critic ~to optimality each gen step
lr = 5e-5                                    # small step; RMSProp (no momentum)

# Generator: DCGAN transpose-conv body, z -> 64x64x3 image. (the implicit sampler g_theta)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co, k=4, s=2, p=1):
            return [nn.ConvTranspose2d(ci, co, k, s, p, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(True)]
        self.net = nn.Sequential(
            *block(nz, ngf*8, 4, 1, 0), *block(ngf*8, ngf*4),
            *block(ngf*4, ngf*2), *block(ngf*2, ngf),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())
    def forward(self, z):
        return self.net(z)

# Critic f_w: DCGAN conv body, NO sigmoid -- it scores, it does not classify.
# Output is a single real number (the dual potential evaluated and averaged).
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co, bn=True):
            layers = [nn.Conv2d(ci, co, 4, 2, 1, bias=False)]
            if bn: layers.append(nn.BatchNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.net = nn.Sequential(
            *block(nc, ndf, bn=False), *block(ndf, ndf*2),
            *block(ndf*2, ndf*4), *block(ndf*4, ndf*8),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False))   # no sigmoid
    def forward(self, x):
        return self.net(x).mean(0).view(1)              # scalar critic value

G, D = Generator(), Critic()
# RMSProp, not Adam: the critic loss is nonstationary; momentum made it diverge.
optD = optim.RMSprop(D.parameters(), lr=lr)
optG = optim.RMSprop(G.parameters(), lr=lr)

def enforce_constraint(net):
    for p in net.parameters():
        p.data.clamp_(-c, c)

def scoring_network_loss(net, real, fake):
    return net(fake) - net(real)             # minimize this == maximize E[f(real)] - E[f(fake)]

def generator_loss(net, fake):
    return -net(fake)                        # raise the critic score assigned to generated samples

def set_requires_grad(net, flag):
    for p in net.parameters():
        p.requires_grad = flag

def train_step(real, gen_iter):
    # Critic phase: estimate W by maximizing E[f(real)] - E[f(fake)].
    real = real[0] if isinstance(real, (list, tuple)) else real
    device = real.device
    set_requires_grad(D, True)
    iters = 100 if (gen_iter < 25 or gen_iter % 500 == 0) else n_critic
    lossD = None
    enforce_constraint(D)
    for _ in range(iters):
        z = torch.randn(real.size(0), nz, 1, 1, device=device)
        fake = G(z).detach()                    # critic phase: generator frozen
        lossD = scoring_network_loss(D, real, fake)
        optD.zero_grad(); lossD.backward(); optD.step()
        enforce_constraint(D)                   # Lipschitz constraint via weight clipping

    # Generator phase: descend W via grad_theta W = -E[grad_theta f(g(z))].
    set_requires_grad(D, False)
    z = torch.randn(real.size(0), nz, 1, 1, device=device)
    fake = G(z)
    lossG = generator_loss(D, fake)
    optG.zero_grad(); lossG.backward(); optG.step()
    # -lossD = E[f(real)] - E[f(fake)] is, up to scale K, the current W estimate.
    return (-lossD).item()
```

The chain is now tight: real images and any implicit generator both live on thin manifolds that generically don't overlap, so every density-ratio distance — KL, JS, every f-divergence — is flat or infinite there and its gradient dies, which is the root of GAN instability, vanishing generator gradients, and the fixed-discriminator collapse mechanism. The Wasserstein-1 / Earth-Mover distance measures transport cost using the metric on the space, so it stays continuous and a.e. differentiable across non-overlapping supports (parallel lines: W = |theta|, gradient sign(theta), while JS = log 2 is flat). Its intractable inf-over-couplings becomes, by Kantorovich–Rubinstein duality, a sup over 1-Lipschitz functions, which I realize with a neural critic that scores rather than classifies; I enforce the Lipschitz constraint crudely by clipping the critic's weights into a small box, train the critic toward optimality because a better critic gives a better W-gradient, and descend `-D(fake)` through the generator with RMSProp.

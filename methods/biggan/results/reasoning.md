Let me start from the gap, because the gap is embarrassing. The best class-conditional ImageNet generator I have produces samples that score about 52 on Inception Score; real ImageNet data scores about 233. Stare at that for a second. The model is more than four times short of real images on a metric that is itself fairly forgiving — IS doesn't even penalize lack of variety within a class, it just wants each image to look confidently like *some* class and the marginal over classes to be spread out. So I'm losing badly on a generous yardstick. And when I look at the samples, the failure is in both directions at once: the good-looking ones are good-looking because they're stereotyped, and when I push for diversity the fidelity falls apart. There's no dial. I'd like a dial.

Why is this so hard? Supervised image classifiers got better more or less by getting bigger — more data, more layers, more channels, and the loss keeps going down. Adversarial training doesn't obviously work that way. The thing I'm optimizing isn't a loss with a floor; it's a two-player game, min_G max_D E_x[log D(x)] + E_z[log(1−D(G(z)))], whose intended resting point is a Nash equilibrium, not a minimum. The dynamics are coupled and famously twitchy: change the architecture, the optimizer, the relative learning rates, and the whole thing can diverge. So the first real question isn't "what clever module do I add," it's blunter: does this game even survive being scaled up, and if not, why not?

Let me set the toolbox out, because I'm not building from nothing. I have a ResNet-style convolutional G and D. I have spectral normalization, which divides each weight matrix by its largest singular value σ₀(W) — estimated cheaply by a single power-iteration step that reuses the left singular vector from last time — so that each layer is 1-Lipschitz and the discriminator can't produce arbitrarily steep, exploding gradients. The point of bounding D's Lipschitz constant is that D then hands G a bounded, usable gradient *everywhere* instead of a spiky one. Prior work found that putting spectral norm in G too, not just D, helps stability and lets me take fewer D steps. I have the non-local self-attention block: a softmax over pairwise feature similarities so each spatial location can pull from every other location, which fixes the thing stacked 3×3 convolutions are bad at — long-range structure, the overall layout of an object rather than its local texture. I have the hinge objective: D pushes real scores above +1 and fake below −1, with losses E[relu(1−D(x))] + E[relu(1+D(G(z)))] for D and −E[D(G(z))] for G; it plays nicely with spectral norm because once the output scale is constrained, the margin actually means something. And I have two ways to tell the model the class.

The class-conditioning question is worth slowing down on, because the obvious thing is wrong. The obvious thing — AC-GAN style — is to concatenate a one-hot class vector to z at the input and bolt an auxiliary classifier onto D. But the auxiliary classifier rewards samples that are easy to classify, and "easy to classify" is exactly "stereotyped," so it actively pushes against variety. That's the wrong incentive for the problem I have. A cleaner channel into G is through normalization: instead of BatchNorm's single learned gain and bias, produce per-class gain γ(c) and bias β(c) and modulate the normalized features as BN(h)·γ(c) + β(c). The class doesn't fight for space in the activations; it just sets the affine knobs of every normalization layer. That's how the class gets into G.

For D, I want to know the principled way to use the label, so let me actually derive it instead of guessing. The optimal discriminator for this game is a sigmoid of the log density ratio: D*(x,y) = σ(f(x,y)) with f(x,y) = log q(x,y)/p(x,y), where q is data and p is the generator's joint. Factor the joint as conditional times marginal: f = log q(y|x)/p(y|x) + log q(x)/p(x). Now suppose the class posteriors, for both the real and generated distributions, are log-linear in some shared feature φ(x) — that is, q(y=c|x) ∝ exp(v_c·φ(x)) and likewise for p, which is just saying "a linear softmax classifier on top of features is a reasonable model of p(class|image)." Then the conditional log-ratio is log q(y|x) − log p(y|x) = (v_y^q − v_y^p)·φ(x) plus the difference of the two log-partition functions, and those partition terms depend only on x. Fold the per-class difference vector v_y^q − v_y^p into a single learned embedding row V[y], and dump everything that depends on x alone — the partition terms and the marginal ratio log q(x)/p(x) — into one scalar function ψ(φ(x)). What drops out is

  f(x,y) = yᵀ V φ(x) + ψ(φ(x)).

The label enters as an *inner product* between a learned class embedding and the discriminator's feature vector, added on top of an ordinary unconditional critic. No concatenation, no auxiliary classifier — the class shows up exactly where the probabilistic model says it should. In code this is just: pool D's features to a vector h, compute an unconditional score linear(h), and add ⟨embed(y), h⟩. I'll use this.

So my baseline is all of that assembled: ResNet G/D, attention in both, spectral norm in both, hinge loss, conditional BatchNorm in G, projection in D, different learning rates for G and D. Now the actual experiment — scale it.

Start with the batch. I increase it by a factor of eight. I expected a modest gain and a lot of pain; instead the IS jumps by almost half, immediately, just from the larger batch. Why would batch size alone do that, when in supervised learning a bigger batch mostly buys you a steadier gradient and you have to retune the learning rate to cash it in? Here it's because each minibatch is a sample of the *modes* of the data, and the gradient each network gets is an expectation over that sample. ImageNet has a thousand classes and enormous intra-class diversity; a small batch sees a thin, high-variance slice of that, so both players are chasing a noisy, mode-starved estimate of the game. Eight times the batch covers many more modes per step, so both the discriminator's notion of "real" and the generator's gradient toward it are far less biased. The networks also reach better final quality in fewer iterations. Good.

But — and this is the side effect that's going to dominate everything — the bigger-batch models become unstable. They climb faster and then undergo *complete training collapse*: sample quality falls off a cliff over a few hundred iterations. For now I do the unglamorous thing and checkpoint just before collapse, because I want to keep measuring the benefits of scale while I figure out the instability separately.

Next, width. I add 50% more channels everywhere, which roughly doubles the parameter count, and IS climbs another fifth or so. That one I believe straightforwardly: the dataset is complex and the model was capacity-limited relative to it, so more channels means more room to represent it. I try the other obvious axis, depth — double it by stacking an extra residual block after each up/down block — and it doesn't help; it hurts. (I'll come back to depth much later with a different, bottleneck-style block; for now, width is the lever, not naive depth.)

Now two architecture changes that aren't about raw size but about *how the conditioning is wired*, and both of them I can motivate from a cost I can see. The class-conditional BatchNorm needs, per layer, a vector of class embeddings to project into gains and biases. With many conditional-BN layers, each holding its own embedding table over a thousand classes, that's a mountain of weights doing nearly-redundant work. So share one class embedding and *linearly project* it to each layer's gains and biases. Fewer parameters, less memory, and — the part I didn't fully anticipate — it reaches a given quality in about a third fewer iterations. I think sharing forces a single, coherent class representation that every layer reads from, instead of a thousand little tables drifting independently.

The second wiring change starts from a question about z. I'm feeding the latent only into the very first layer; from there on the network has to carry whatever it needs about z up through every resolution. But z is supposed to be the handle on *all* the factors of variation, and those factors live at different scales — pose and layout are coarse, texture is fine. Why force all of that to squeeze through the bottom layer and survive the climb? Let me give the generator direct access to z at every resolution. Concretely, split z into equal chunks, one per residual block, and at each block concatenate its chunk to the shared class embedding before projecting to that block's BN gains and biases. Now the latent can directly modulate features at every level of the hierarchy. It buys a few percent of quality and another chunk of training speed. (Splitting also trims the first linear layer, since the bottom block only consumes its own chunk of z.) I'll call this giving z skip connections into the blocks.

That settles the scaling and conditioning story. Now the dial I said I wanted — the fidelity/variety knob.

Here's a freedom that GANs have and most generative models don't. A VAE or a normalizing flow has to *backpropagate through its latents* or evaluate a density in latent space, so its prior is load-bearing and you're stuck with it. A GAN never inverts z; G just consumes whatever z I hand it. So I'm free to choose the prior — and, more to the point, I'm free to sample z at test time from a *different* distribution than the one I trained with. I trained with z ∼ N(0, I). What if at test time I draw z from a *truncated* normal — resample any coordinate whose magnitude exceeds some threshold so it lands back inside? Try it. The IS and FID both improve immediately, just from changing the sampling distribution of a fixed, already-trained model.

Why on earth would that help? Think about where the generator is actually any good. During training, z came from N(0, I), so the network saw latents overwhelmingly from the high-density core of the Gaussian and almost never from the far tails. It's best-fit exactly where it got the most supervision: near the mode of the prior. The tails are the under-trained regions, the places where G has to extrapolate. Truncating throws away the tails and keeps z near the high-density core, so every sample comes from a region the generator models well — fidelity goes up. And as I shrink the threshold toward zero, z collapses toward the prior's mode, so the outputs collapse toward the mode of G's per-class distribution: the single most canonical image for that class. Variety goes down. So truncation is a continuous slider between "diverse but riskier" and "canonical but high-fidelity," and crucially it's *post-hoc* — one trained model, a dial I turn at sampling time.

I can even read the slider as a curve. Sweep the threshold and plot IS against FID and it looks like a precision/recall curve. IS, which doesn't punish missing intra-class variety, behaves like precision: it just keeps rising as I truncate harder. FID punishes both bad fidelity and dropped variety, so it behaves like a mix: it improves at first as fidelity climbs, then turns and dives as truncation eats the variety. That's a clean, useful picture of the trade-off, and it's free.

Except it doesn't work on every model. Some of my larger generators, fed truncated z, don't sharpen — they produce saturation artifacts, blown-out garbage. Stare at why. Feeding z from near the mode is feeding the network inputs from a region it didn't see densely in training — a distribution shift — and the only way truncation is safe is if G is *smooth*: if nearby latents map to nearby, sensible images, so that the whole z-space, not just the cloud of training samples, maps to good outputs. A jagged, ill-conditioned G will have pockets where a perfectly reasonable truncated z lands on nonsense. So the truncation trick is really demanding a property of G — smoothness, good conditioning — and I haven't been enforcing it. If I want truncation to be reliable, I have to *make* G smooth.

How do I enforce smoothness? I want the linear maps inside G to be well-conditioned, not stretching some directions enormously while crushing others — that kind of anisotropy is exactly what makes a small change in z explode into a large change in output. The clean way to ask a weight matrix to be norm-preserving is to push it toward orthonormality. Treat each output filter as a row of `W`, so the relevant Gram matrix is `WWᵀ`. The standard orthogonal regularizer is

  R_β(W) = β‖WWᵀ − I‖²_F,

which says: make the Gram matrix of the filters the identity — unit norm and mutually orthogonal. I try it, and it's too blunt. Forcing WWᵀ = I constrains the *norms* of the filters as well as their directions, and the network has legitimate reasons to want filters of varying magnitude; clamping all of that strangles it. So relax. What I actually care about for conditioning is that the filters point in *different* directions — that they're decorrelated — not that they all have unit length. The off-diagonal entries of WWᵀ are the pairwise inner products of the filters; the diagonal entries are their squared norms. So drop the diagonal and penalize only the off-diagonal part:

  R_β(W) = β‖WWᵀ ⊙ (1 − I)‖²_F,

where (1 − I) is the all-ones matrix with the diagonal zeroed. This minimizes the pairwise cosine-similarity-like cross terms between filters — decorrelating their directions — while leaving their norms completely free. That's the smoothness I wanted without the strangling. I don't even need to materialize the loss and differentiate it; the gradient of this penalty is just 2·(WWᵀ ⊙ (1 − I)) W, so I add strength · that directly to the weight's gradient each step (the full-orthogonal version is the same with (WWᵀ − I) in place of WWᵀ ⊙ (1 − I)). I sweep the strength over a few orders of magnitude and settle on 1e-4 — a tiny nudge. And the payoff is exactly the one I was after: without this regularization only about a sixth of my models are amenable to truncation; with it, well over half are. The penalty isn't there to boost the metric directly; it's there to *make the truncation dial usable*. The two pieces — sampling near the prior's core, and conditioning G so that's safe — only make sense together.

Now I have to face the collapse, because I've been papering over it with early stopping and that's not an answer. These instabilities show up at scale even though the very same settings are stable at small scale, so toy analyses won't cut it; I have to look directly at the large-scale runs. I monitor everything — weight, gradient, and loss statistics — hunting for something that *precedes* collapse. The most informative quantities turn out to be the top few singular values of each weight matrix, σ₀, σ₁, σ₂, which I can track with an extended power iteration (the same machinery as spectral norm, pushed to recover a couple more singular vectors).

Look at G first. Most layers have well-behaved spectra. But a few — typically the very first layer, the over-complete non-convolutional linear that turns z into a 4×4 feature map — misbehave: their σ₀ grows steadily throughout training and then *explodes* right at collapse. That's a strong lead. Is the explosion the *cause* of collapse, or just a symptom that rides along with it? I have to test causation, not correlation. So I attack σ₀ directly. First, regularize it: penalize σ₀ toward a fixed target, or toward a ratio of the second singular value, r·sg(σ₁), with a stop-gradient on σ₁ so the penalty can't cheat by inflating σ₁ instead of shrinking σ₀. Alternatively, *clamp* it surgically with a partial SVD — take the top singular triple (σ₀, u₀, v₀) and subtract off the excess:

  W ← W − max(0, σ₀ − σ_clamp) v₀ u₀ᵀ,

with σ_clamp set to that fixed value or to r·sg(σ₁). Both interventions do exactly what they promise: they stop σ₀ (or the ratio σ₀/σ₁) from creeping up and exploding, with or without spectral norm layered on top. And yet — collapse still happens. Sometimes performance even nudges up a little, but nothing I do to G's spectrum prevents the crash. So the σ₀ explosion in G is a symptom, an indicator I can watch, not the lever. Conditioning G is, at best, necessary; it is plainly not sufficient. The disease isn't purely in G.

Turn to D. Its spectra look different: noisier, with σ₀/σ₁ sitting near one (a slow spectral decay, weights close to orthogonal), and σ₀ grows through training but only *jumps* at collapse rather than smoothly exploding. The Frobenius norms, meanwhile, stay smooth — so whatever the noise is, it's concentrated in the top singular directions, not spread across the whole matrix. Up close, each spike looks like an impulse response: a sudden jump, then a slow decaying oscillation back down. The natural reading is that this is the adversarial dynamics biting: every so often G produces a batch that strongly perturbs D along its leading directions.

If that spectral noise is what's destabilizing things, the textbook counter is to regularize D's Jacobian directly — a zero-centered gradient penalty on real data, R₁ = (γ/2) E_{q_data}‖∇D(x)‖²_F. I turn it on at the suggested strength γ = 10, and training becomes stable: the spectra of both networks get smoother and bounded, no more collapse. And the model is *much worse* — IS drops by nearly half. I dial the penalty down; performance recovers as I weaken it, but the spectra get uglier, and below a strength of about 1 the sudden collapse comes back. Even at γ = 1, the lowest strength that still prevents collapse, I'm paying about a fifth of my IS for the privilege. I try the same bargain with other constraints on D — orthogonal reg, dropout in D's final features, L2 — and they all tell the same story: crank the penalty hard enough and you buy stability, every time, at a steep and unavoidable cost in quality. Stability is purchasable, but the currency is performance.

There's a second clue in D that explains a lot. D's loss drifts toward zero through training and then jumps sharply up at collapse. Is D *overfitting*? I run the obvious test: evaluate an un-collapsed discriminator on the held-out ImageNet validation set and ask how many images it classifies correctly as real or fake. On the training set it's above 98%. On validation it's 50–55% — coin-flip, no matter what regularization I use. So D is flatly memorizing the training set; it learns no generalizing boundary between real and fake. At first that sounds like a bug, but think about D's actual job: it isn't supposed to generalize, it's supposed to distill the training data into a useful gradient signal for G. Memorization is consistent with that role. And memorization actually *explains the spikes*: as D approaches perfect memorization of the reals, it gets less and less signal from them, because both the hinge loss and the original log loss give exactly *zero gradient* on an example D already scores confidently and correctly. With the real-data gradient attenuating toward zero, D keeps receiving the fake-side gradient that pushes its outputs negative, so it slowly drifts toward a negative output bias — until that bias gets large enough that D starts misclassifying a batch of *real* images, takes a big corrective gradient pushing outputs positive, and that's the impulse spike I saw. That story even suggests fixes: an unbounded loss like Wasserstein wouldn't have the zero-gradient-on-correct problem (but it trains unstably here even with gradient penalties); widening the hinge margin pulls more examples inside the margin so they keep contributing gradient (but a smaller margin hurts, a larger margin up to 3× doesn't fix collapse, and beyond 3× it goes unstable like Wasserstein); shrinking D to limit memorization (but that degrades training). None of them is a clean win.

So is it G's fault or D's? I run the cleanest experiment I can think of: freeze one network and let the other keep training. Freeze G, keep training D — D stays perfectly stable and quietly drives its loss toward zero. Freeze D, keep training G — G collapses immediately and catastrophically, driving D's loss up past 300 against a normal range of about 0 to 3. The asymmetry is the whole point. Given a fixed, frozen D, G instantly finds adversarial inputs that blow the game apart; D's gradients are only valid for the *instantaneous* G that produced them. So D has to stay optimal with respect to G at every step, or G runs away. But — and this is the part that closes the loop with the R₁ experiments — favoring D harder (bigger D learning rate, more D steps per G step) does *not* buy stability either; past a couple of D steps it just makes things worse. Stability isn't a property of G's conditioning, and it isn't a property of D's conditioning; it's a property of their *interaction* through the adversarial game. I can use the singular-value symptoms to see trouble coming, and good conditioning is necessary for training to proceed, but it is not sufficient to prevent the eventual crash.

That leaves me with an honest, slightly uncomfortable engineering conclusion rather than a tidy fix. I *can* enforce stability — strongly constrain D and the crash goes away — but every knob that does so costs a large, irreducible chunk of performance. The better trade in practice is the opposite: keep the conditioning light, accept that collapse will come late in training, and stop just before it, by which point the model is already very good. Scale, the right conditioning, and the truncation dial get me the quality; early stopping handles the collapse I can characterize but can't cheaply prevent.

Let me write it down, grounded in what actually trains. The discriminator's projection output, the conditional-BN with shared embedding and skip-z, the modified orthogonal penalty as a direct gradient, the hinge losses, the EMA, and the truncated sampler:

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Class-conditional BatchNorm: one shared embedding, projected per layer to
# (gain, bias). gain centered at 1, bias at 0. cond = [class_embed ; z_chunk].
class ConditionalBN(nn.Module):
    def __init__(self, num_features, cond_dim, which_linear):
        super().__init__()
        self.gain = which_linear(cond_dim, num_features, bias=False)
        self.bias = which_linear(cond_dim, num_features, bias=False)
        self.bn   = nn.BatchNorm2d(num_features, affine=False)  # cross-replica / standing in practice
    def forward(self, x, cond):
        gain = (1 + self.gain(cond)).view(x.size(0), -1, 1, 1)
        bias = self.bias(cond).view(x.size(0), -1, 1, 1)
        return self.bn(x) * gain + bias

# Generator residual block: conditioned on cond = [shared_class_embed ; this block's z chunk].
class GResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, which_conv, which_bn):
        super().__init__()
        self.bn1, self.bn2 = which_bn(in_ch, cond_dim), which_bn(out_ch, cond_dim)
        self.conv1 = which_conv(in_ch, out_ch)
        self.conv2 = which_conv(out_ch, out_ch)
        self.learnable_sc = (in_ch != out_ch)
        if self.learnable_sc:
            self.conv_sc = which_conv(in_ch, out_ch, kernel_size=1, padding=0)
    def forward(self, x, cond):
        h = F.relu(self.bn1(x, cond))
        h = F.interpolate(h, scale_factor=2)            # nearest-neighbour upsample
        x = F.interpolate(x, scale_factor=2)
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h, cond)))
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x

class Generator(nn.Module):
    def __init__(self, dim_z=120, n_classes=1000, ch=96, shared_dim=128):
        super().__init__()
        which_conv  = SNConv2d                          # spectral norm in G
        which_lin   = SNLinear
        self.n_slots = 5                                # one z chunk per res block (128px)
        self.chunk   = dim_z // (self.n_slots + 1)      # 20-D chunks
        cond_dim     = shared_dim + self.chunk          # class embed concatenated with a z chunk
        self.shared  = nn.Embedding(n_classes, shared_dim)   # ONE shared class embedding
        self.linear  = which_lin(self.chunk, 16*ch * 4*4)    # first layer takes only its chunk
        which_bn = lambda c, d: ConditionalBN(c, d, which_lin)
        chans = [16*ch, 16*ch, 8*ch, 4*ch, 2*ch, ch]
        self.blocks = nn.ModuleList(
            [GResBlock(chans[i], chans[i+1], cond_dim, which_conv, which_bn)
             for i in range(self.n_slots)])
        self.attn = SelfAttention(2*ch, which_conv)     # one non-local block at 64x64
        self.attn_after = 3                             # after the block whose output is 2*ch
        self.out  = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(),
                                  which_conv(ch, 3))
    def forward(self, z, y):
        zs = torch.split(z, self.chunk, 1)              # skip-z: split z into per-block chunks
        emb = self.shared(y)
        conds = [torch.cat([emb, zs[i+1]], 1) for i in range(self.n_slots)]
        h = self.linear(zs[0]).view(z.size(0), -1, 4, 4)
        for i, blk in enumerate(self.blocks):
            h = blk(h, conds[i])
            if i == self.attn_after:                    # attention after the 64x64 block
                h = self.attn(h)
        return torch.tanh(self.out(h))

class Discriminator(nn.Module):
    def __init__(self, n_classes=1000, ch=96):
        super().__init__()
        which_conv = SNConv2d
        self.blocks = nn.ModuleList([...])              # ResBlock-down stack + SelfAttention at 64x64
        self.linear = SNLinear(16*ch, 1)
        self.embed  = SNEmbedding(n_classes, 16*ch)     # projection embedding
        self.activation = nn.ReLU()
    def forward(self, x, y):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = torch.sum(self.activation(h), [2, 3])        # global sum pooling -> feature vector
        out = self.linear(h)                             # unconditional critic  psi(phi(x))
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)   # + <V[y], phi(x)>  (projection)
        return out

# Hinge objective
def loss_hinge_dis(d_fake, d_real):
    return F.relu(1. - d_real).mean(), F.relu(1. + d_fake).mean()
def loss_hinge_gen(d_fake):
    return -d_fake.mean()

# Modified orthogonal regularization, applied as a direct gradient on G's weights.
# grad of beta * || W Wᵀ ⊙ (1 - I) ||²_F   is   2 (W Wᵀ ⊙ (1 - I)) W.
def ortho(model, strength=1e-4, blacklist=()):
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim < 2 or any(p is b for b in blacklist):
                continue                                 # skip the shared embedding
            w = p.view(p.shape[0], -1)
            grad = 2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w)
            p.grad.data += strength * grad.view(p.shape)

# Truncation trick: at sampling time draw z from a truncated normal, then scale.
# Smaller `truncation` -> z nearer the prior's mode -> higher fidelity, less variety.
from scipy.stats import truncnorm
def truncated_z_sample(batch, dim_z, truncation=0.5):
    values = truncnorm.rvs(-2, 2, size=(batch, dim_z))   # standard normal, clipped to [-2, 2]
    return truncation * values

def train_step(G, D, GD, real_x, real_y, z, y, ema, cfg):
    for _ in range(cfg['num_D_steps']):                  # D must keep up with G: 2 steps per G step
        D.optim.zero_grad()
        d_fake, d_real = GD(z.sample_(), y.sample_(), real_x, real_y, train_G=False)
        d_loss_real, d_loss_fake = loss_hinge_dis(d_fake, d_real)
        (d_loss_real + d_loss_fake).backward()
        D.optim.step()
    G.optim.zero_grad()
    d_fake = GD(z.sample_(), y.sample_(), train_G=True)
    loss_hinge_gen(d_fake).backward()
    ortho(G, cfg['G_ortho'], blacklist=list(G.shared.parameters()))   # light smoothness penalty
    G.optim.step()
    ema.update()                                         # average G's weights for sampling
```

The causal chain, start to finish: the gap to real images is huge and adversarial training doesn't scale for free, so I scale anyway — eight times the batch (more modes per gradient) and half again the width (more capacity for a complex dataset) — and quality leaps but training starts collapsing; I rewire the conditioning to be cheaper and richer (one shared class embedding projected to every conditional-BN layer, plus z fed into every block) which speeds training and lifts quality further; because a GAN never inverts its latent I'm free to sample z from a truncated normal at test time, trading variety for fidelity by staying in the high-density region where G is best modeled, which gives me the dial I wanted — but the dial is only safe if G is smooth, so I add a light off-diagonal orthogonal penalty that decorrelates filter directions without constraining their norms, making far more models amenable to truncation; and when I dissect the collapse by watching the top singular values, I find G's σ₀ explosion is a symptom I can suppress but not a cause, D memorizes the data and its zero-gradient-on-correct loss drives the spectral spikes, and freezing experiments show stability lives in the G–D interaction — purchasable by heavily constraining D, but only at a steep performance cost, so the practical answer is to keep conditioning light and stop training just before the late collapse.

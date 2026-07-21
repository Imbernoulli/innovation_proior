The conditioning operator is the whole task, and before I can compare operators I need a floor — the
first concrete fill of `prepare_conditioning` plus `ClassConditioner` that turns the scaffold's
unconditional UNet into a class-conditional one. I want to start from the operator whose *cost and
structure are the tightest match to what the conditioning signal actually is*, so that whatever it leaves
on the table is a clean diagnosis of what the next step must add.

What I have is a noise-prediction UNet, $\epsilon_\theta(x_t, t)$: feed it a noised CIFAR image and the
timestep, it predicts the noise. The training loss is the friendly one — draw $x_0$, a timestep $t$, a
noise vector $\epsilon$, form $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ in one
shot, regress $\lVert \epsilon - \epsilon_\theta(x_t,t)\rVert^2$. The third argument is what I'm adding:
a class index $c$, so the same noised latent at the same noise level denoises differently toward a
"horse" than toward a "ship." Crucially the backbone already conditions on *one* scalar side input — the
timestep $t$ — and it does so through the diffusers UNet's own machinery: $t$ becomes a sinusoidal
embedding, the time-embedding MLP lifts it to `time_embed_dim`, and inside every residual block that
vector drives the block's adaptive normalization (the diffusers `ResnetBlock2D` consumes `temb` and turns
it into a per-channel scale/shift on the block's GroupNorm). So the UNet has a *finished, tuned socket*
for a global per-channel side signal. The class label is exactly that shape — one global per-example
vector — so the question I must answer first is how that signal should be carried.

Let me characterize the class signal precisely, because the right operator follows from its structure. A
class index has no spatial layout — there is no "where" to a label, it applies to the whole image — and
no internal structure: it is one categorical value, which I embed into one vector. It is *global* (the
same for every spatial position), *per-channel* (its natural action is "turn some feature channels up,
others down"), and *content-blind* in the sense that it is drawn from the label alone, not from what is
currently forming in the feature map. That is the same profile as the timestep. So I cost the operator
families the prior art hands me against that profile. Concatenation would broadcast the class vector into a
constant $[B, d, H, W]$ slab and stack it on the input before a conv — but a label has nothing image-shaped
to concatenate, so I would be manufacturing $H\times W$ copies of one vector for a conv to then learn to
ignore, and forcing the signal in at the input only, where the convolutional stack must transport it
through every resolution rather than re-injecting it. Bounded gating, $F \mapsto \sigma(g(z))\odot F$ with
$\sigma\in(0,1)$, can only *attenuate* channels — no shift, no amplification past the current magnitude,
and $\sigma$ saturates. Attention treats the class as a token to read from position by position with
content-dependent weights, but here the set has one element, so its weighting has nothing to weigh and most
of the machinery sits idle — and it is the most expensive of the four. That leaves the operator that takes
the conditioning vector and *modulates every channel of every spatial location in the same way*: an
adaptive affine — global, per-channel, location-agnostic, with nothing idle for a single label.

This is the FiLM primitive, and the conditioning lineage converged on it from several directions, which
is why I trust it as the floor. FiLM states it generally: a conditioning input $z$ regresses a
per-channel scale and shift, $\mathrm{FiLM}(F) = \gamma(z)\cdot F + \beta(z)$, applied wherever there is a
feature map, with parameter count tied to channels (not resolution) so it is cheap to apply repeatedly.
Concatenation-then-linear reduces to the $\gamma = 1$ corner ($W[F;z] = W_F F + W_z z$, an additive
conditional bias); bounded gating is the $\beta = 0$, squashed-$\gamma$ corner. Conditional BatchNorm,
Conditional Instance Norm, and AdaIN are all the same affine attached to one normalization or another.
And most on-the-nose for a denoiser, ADM's adaptive Group Norm is exactly this affine living inside the
residual block, driven by the combined timestep-and-class embedding — and that recipe ablated it against
the older "add the embedding, then normalize" move and found the adaptive affine better. So the
established answer to "how do I inject a global side signal into a diffusion residual block" is "modulate
a normalization's affine from it." That is the family I will draw the floor from.

Now I have to make two design decisions that the FiLM family leaves open, and these are where the floor
earns or loses its keep. First: *where* does the class signal enter — the time path
(`prepare_conditioning`), the post-block path (`ClassConditioner`), or both? The scaffold gives me two
sockets. The path-of-least-resistance is to fold the class into the time embedding and let the UNet's own
in-block AdaGN carry it — that is the additive `time_emb + class_emb` move, and it is the road I mean to
try *next*, once I have a floor to measure it against, so I will deliberately not take it here. Instead I
want the floor to be a *self-contained, post-block adaptive-norm conditioner*: leave `prepare_conditioning`
an identity (`return time_emb`, so the timestep keeps its own tuned path untouched and the block AdaGN
stays a pure noise-level signal), and put *all* the class conditioning into the `ClassConditioner` module
applied after each block, as an explicit adaptive normalization on the `[B,C,H,W]` feature map. The
class never touches the time embedding; it acts only through a dedicated affine modulation of the
post-block features. This makes the floor a clean test of the post-block adaptive-norm operator on its
own — one thing conditioned in one place, so whatever it fails to buy is unambiguous.

Second decision, and the one I want to nail because it bites at initialization: the affine's
parameterization and its starting point. The naive form $\gamma(z)\cdot F + \beta(z)$ with a freshly
initialized regressor sits near $\gamma \approx 0, \beta \approx 0$, which *annihilates* the feature map
on step one: the modulated feature is $0\cdot F + 0 = 0$, and the local derivative through that path is
$\gamma \approx 0$ too, so the gradient is nearly killed with it — the layer is not just wrong at init but
barely gradient-able through its main path. The identity affine is not $\gamma = 0$; it is $\gamma = 1,
\beta = 0$. So I parameterize the scale as a *deviation from one*: write the modulation as
$(1+\text{scale})\cdot \hat F + \text{shift}$ where $\hat F$ is the normalized feature, so that
$\text{scale}=\text{shift}=0$ leaves the normalized feature unchanged and the regressor learns only how far
to push *away* from "no modulation." Same expressible set of affines, a sane origin instead of a
destructive one.

But there is a deeper initialization concern, because I am inserting a *brand-new* conditioning sublayer
into the middle of a backbone whose weights and learning-rate schedule were tuned for a denoiser. If I
drop a randomly-initialized affine modulator after each block, at step zero it injects a random
perturbation into a carefully balanced residual stream — I would be throwing away the good initialization
and forcing training to first undo the damage. The principled fix is the zero-init residual trick
(zeroing the last scale of a residual branch so the branch starts as the identity — Goyal et al. 2017's
last-BN-$\gamma$-zero, and the diffusion UNet's zero-last-conv). I want the inserted conditioner to be the
*exact identity at init* and to grow its effect only as training finds a use for it. The clean way is a
per-channel *gate* on the modulated-minus-original difference, regressed from the class like scale and
shift, positioned right at the residual add, and *zero-initialized*. With $\text{gate} = 0$ the module
returns its input bit-for-bit; the network is the original denoiser at init; and as the gate moves off
zero the class conditioning ramps up smoothly. This is adaptive normalization, zero-initialized —
adaLN-Zero — but rendered here on a convolutional feature map rather than on a transformer token stream.

The provided `AdaLNBlock(channels, cond_dim)` is precisely this: a GroupNorm of the feature map, then a
zero-initialized `SiLU → Linear(cond_dim, 3·channels)` emitting scale, shift, and gate, applied as
$$x \;+\; \text{gate}\cdot\big((1+\text{scale})\cdot\mathrm{norm}(x) + \text{shift} - x\big).$$
With the whole regressor zero-initialized, at init $\text{scale}=\text{shift}=\text{gate}=0$ and the update
is $x + 0\cdot((1+0)\hat x + 0 - x) = x$: the exact identity at every element, so the network is
bit-for-bit the tuned denoiser and the good init is preserved. The gradient at init is where I have to be
careful, because a zero-init module could be a *dead* one. Write the branch as $o = x + g\,\big((1+s)\hat x
+ b - x\big)$ with $g, s, b$ the gate/scale/shift from the zero-init Linear on $\phi = \mathrm{SiLU}(
\text{class emb})$. Then $\partial o/\partial g = (1+s)\hat x + b - x$, which at init is $\hat x - x \neq 0$;
but $\partial o/\partial s = g\,\hat x = 0$ and $\partial o/\partial b = g = 0$. Back at the Linear, only
the *gate* row receives gradient at init; scale and shift stay frozen until the gate moves off zero. That
is a two-phase dynamic: first the gate learns how much of the $\hat x - x$ direction to blend in, then
scale and shift wake up and the modulation gains real per-channel structure. So the floor does not
condition instantly — it spends early training climbing the gate off zero, a cost paid per inserted
conditioner, and I will remember it when I read where this operator underperforms.

The normalization choice carries an inductive-bias decision. The `AdaLNBlock`'s norm is `GroupNorm(1)` —
a single group over all $C$ channels — so statistics are pooled across the *entire* channel-and-spatial
extent per sample, giving $\hat x$ a global normalization (LayerNorm-over-$(C,H,W)$ in GroupNorm clothing,
not the diffusers default of many groups). That reinforces the profile I want: the class affine acts on a
globally-normalized feature and pushes it globally, matching the structureless label. The scale/shift/gate
come out $[B, C]$ each and broadcast to $[B, C, 1, 1]$ — one number per channel, none per position, which
is the whole point: no spatial address.

One interaction to check, since I am inserting a normalization *right after* a block whose own AdaGN just
wrote a per-channel scale/shift into the feature: does my `GroupNorm(1)` undo it? Inside the branch, yes —
re-normalizing $x$ to $\hat x$ strips the timestep's per-channel gain. But the update carries the leading
$x$ — the block's output with the timestep modulation fully intact — untouched through the residual, and
only the gated *correction* is computed on $\hat x$ (and at init the gate is zero, so the correction is
nil). The timestep's work is never destroyed; the class conditioner adds a class-dependent term on top of
it. That holds only because the operation is residual — a non-residual "normalize then re-modulate" would
have clobbered the timestep signal.

It is also worth being precise about *why* I call this operator low-bandwidth, because it is not a lack of
representational capacity. The affine offers $2C$ degrees of freedom per block (a scale and a shift per
channel), and $2C$ is enormous next to the $\log_2 10 \approx 3.3$ bits needed to name which CIFAR class I
am in — there is no shortage of parameters to *distinguish* the ten classes. The limit is not "can the
affine tell horse from ship" but "can the affine act *differently at different places in the picture*,"
and the answer is structurally no: the same scale and shift hit every spatial position, and neither reads
the local content. So the ceiling I am predicting is an *addressing* ceiling, not a capacity ceiling, and
that distinction is what tells me the fix has to change the operator's spatial/content access rather than
just give it more parameters.

The budget enforces $1.05\times$ the cross-attention reference, so I check the cost. Per conditioner it is
the one `Linear(cond_dim, 3C)`: about $3\,\text{cond\_dim}\cdot C$ weights. At Small, `cond_dim` is
$64\times4 = 256$ and the block channels run $\{64,128,128,128\}$, so the widest conditioner is
$\sim 3\cdot256\cdot128 \approx 98\text{k}$ and the whole set of regressors is under a million against a
$\sim$9M backbone. It scales as $O(\text{cond\_dim}\cdot C)$, not the conv's $O(C^2)$, so its share shrinks
at wider scales, and it is comfortably lighter than the cross-attention reference (per-block projections
$\sim 2C^2 + 2\,\text{cond\_dim}\cdot C$). The budget will bind on a heavier operator, not this one.

I should be explicit about what this *is not*, because the same name elsewhere means a much larger thing.
The canonical adaLN-Zero conditions a *transformer* diffusion backbone: it replaces each LayerNorm's affine
in a stack of ViT blocks, sums timestep and class into one vector $c = t_{\text{emb}} + c_{\text{emb}}$,
regresses six modulation vectors per block from a `Linear(d, 6d)`, adds a gateless decode head, and
zero-inits all of it so the whole block stack is the identity at start. None of that machinery is present
here: the backbone is a fixed convolutional UNet I cannot touch, there is no token stream, no decode head,
and I am keeping timestep and class on *separate* paths by choice. What I have is one post-block hook taking
a `[B,C,H,W]` feature map and the class embedding. So I port *only* the core operation — normalize, regress
$(1+\text{scale})\cdot\hat x + \text{shift}$, gate it onto the residual, zero-init the regressor so the
block starts as the identity — onto a convolutional feature map (GroupNorm for LayerNorm), carrying the
class alone.

So the step-1 edit is settled: `prepare_conditioning` returns `time_emb` unchanged, and `ClassConditioner`
wraps one zero-initialized `AdaLNBlock` per block, modulating each post-block feature map from the class
embedding (the full module is in the answer). The training objective is unchanged in shape — the plain
noise-MSE — only the network now sees the class through these adaptive-norm gates; sampling feeds the fixed
class index at every DDIM step.

So the floor should produce a genuinely class-conditional model with a sensible FID at every scale — it
*works* — but I expect it to be the *weakest* operator I try, for two structural reasons. First, I routed
the class only through the post-block path, so it never rides the blocks' own AdaGN the way the timestep
does, and (from the gradient trace) that thin channel must climb the gate off zero before it conditions at
all. Second, the operator is spatially uniform and content-blind — the lowest-bandwidth affine. I cannot
yet put a number on the FID, so I state only the shape: it should fall from Small to Large as bigger
backbones denoise better, and sit *above* whatever a better-routed or richer operator reaches. That points
the next step at a clean fork — a *better route* for the same affine, or a *richer* operator — and the
cheaper thing to try first is the better route.

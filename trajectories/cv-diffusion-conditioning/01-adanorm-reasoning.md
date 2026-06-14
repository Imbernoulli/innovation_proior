The conditioning operator is the whole task, and before I can compare operators I need a floor — the
first concrete fill of `prepare_conditioning` plus `ClassConditioner` that turns the scaffold's
unconditional UNet into a class-conditional one. I want to start from the operator whose *cost and
structure are the tightest match to what the conditioning signal actually is*, so that whatever it leaves
on the table is a clean diagnosis of what the next rung must add. So let me reason carefully about what a
class label *is*, in this denoiser, and which mechanism fits it.

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
currently forming in the feature map. That is the same profile as the timestep. So the operator I want is
not attention (which is for reading from a *set* of structured tokens, position by position, with
content-dependent weights — overkill, and partly ill-defined, for a single structureless token), and not
concatenation (a label has nothing image-shaped to stack on the input). The operator that fits a global,
per-channel, location-agnostic signal is the one that takes the conditioning vector and *modulates every
channel of every spatial location in the same way*: an adaptive affine.

This is the FiLM primitive, and the conditioning lineage converged on it from several directions, which
is why I trust it as the floor. FiLM states it generally: a conditioning input $z$ regresses a
per-channel scale and shift, $\mathrm{FiLM}(F) = \gamma(z)\cdot F + \beta(z)$, applied wherever there is a
feature map, with parameter count tied to channels (not resolution) so it is cheap to apply repeatedly.
Concatenation-then-linear reduces to the $\gamma = 1$ corner ($W[F;z] = W_F F + W_z z$, an additive
conditional bias); bounded gating is the $\beta = 0$, squashed-$\gamma$ corner. Conditional BatchNorm,
Conditional Instance Norm, and AdaIN are all the same affine attached to one normalization or another.
And most on-the-nose for a denoiser, ADM's adaptive Group Norm is exactly this affine living inside the
residual block, driven by the combined timestep-and-class embedding — and that paper ablated it against
the older "add the embedding, then normalize" recipe and found the adaptive affine better. So the
established answer to "how do I inject a global side signal into a diffusion residual block" is "modulate
a normalization's affine from it." That is the family I will draw the floor from.

Now I have to make two design decisions that the FiLM family leaves open, and these are where the floor
earns or loses its keep. First: *where* does the class signal enter — the time path
(`prepare_conditioning`), the post-block path (`ClassConditioner`), or both? The scaffold gives me two
sockets. The path-of-least-resistance is to fold the class into the time embedding and let the UNet's own
in-block AdaGN carry it — that is the additive `time_emb + class_emb` move, and it is the *next* rung's
idea, so I will deliberately not take it here. Instead I want the floor to be a *self-contained, post-block
adaptive-norm conditioner*: leave `prepare_conditioning` an identity (`return time_emb`, so the timestep
keeps its own tuned path untouched and the block AdaGN stays a pure noise-level signal), and put *all* the
class conditioning into the `ClassConditioner` module applied after each block, as an explicit adaptive
normalization on the `[B,C,H,W]` feature map. The class never touches the time embedding; it acts only
through a dedicated affine modulation of the post-block features. This makes the floor a clean
test of the post-block adaptive-norm operator on its own.

Second decision, and the one I want to nail because it bites at initialization: the affine's
parameterization and its starting point. The naive form $\gamma(z)\cdot F + \beta(z)$ with a freshly
initialized regressor sits near $\gamma \approx 0, \beta \approx 0$, which *annihilates* the feature map
on step one — and the local derivative through the modulated path is $\gamma \approx 0$ too, so the
gradient is nearly killed as well. The identity affine is not $\gamma = 0$; it is $\gamma = 1, \beta = 0$.
So I parameterize the scale as a *deviation from one*: write the modulation as $(1+\text{scale})\cdot
\hat F + \text{shift}$ where $\hat F$ is the normalized feature, so that $\text{scale}=\text{shift}=0$
leaves the normalized feature unchanged and the regressor learns only how far to push *away* from "no
modulation." That is the right-conditioned learning problem and the sane init.

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

Let me write the operation concretely against the substrate, because the scaffold hands me the exact
building block. The provided `AdaLNBlock(channels, cond_dim)` is precisely this: a GroupNorm of the
feature map, then a zero-initialized `SiLU → Linear(cond_dim, 3·channels)` that emits scale, shift, and
gate, applied as $x + \text{gate}\cdot\big((1+\text{scale})\cdot\mathrm{norm}(x) + \text{shift} - x\big)$.
Read that update: when gate $=0$ it is $x + 0 = x$ (identity at init); as the gate grows it interpolates
toward the modulated normalized feature; the $(1+\text{scale})$ keeps zero-scale neutral; the SiLU lets
the modulation depend nonlinearly on the class embedding. So my `ClassConditioner` is a thin wrapper that
holds one `AdaLNBlock` per block and applies it, and `prepare_conditioning` returns the time embedding
unchanged. That is the whole floor.

I should be explicit about what this *is not*, because the same name elsewhere means a much larger thing.
The canonical adaLN-Zero is the conditioning mechanism of a *transformer* diffusion backbone: it replaces
each LayerNorm's learned affine in a stack of ViT blocks, sums the timestep and class embeddings into one
conditioning vector $c = t_{\text{emb}} + c_{\text{emb}}$, regresses six modulation vectors per block
(scale/shift/gate for the attention sublayer and for the MLP sublayer) from a single `Linear(d, 6d)`,
adds a gateless adaptive-norm decode head, and zero-initializes all of it so the *entire block stack* is
the identity map at start, all to inherit transformer scaling. None of that applies here. The harness has
no transformer — the backbone is a fixed convolutional UNet I am forbidden to touch. There is no token
stream, no patchify, no final decode head to condition, and no $c = t+y$ sum (I am keeping the timestep
and class on *separate* paths by choice). What the harness exposes is exactly one thing: a post-block hook
that takes a `[B,C,H,W]` feature map and the class embedding. So I am porting *only* the core
adaLN-Zero operation — normalize, regress $(1+\text{scale})\cdot\hat x + \text{shift}$, gate it onto the
residual, zero-init the regressor so the block starts as the identity — onto a convolutional feature map
(GroupNorm in place of LayerNorm), inserted after each UNet block, carrying the class signal alone. The
transformer machinery the canonical method is famous for is simply not present, and importing its story
would mis-describe what runs here.

So the step-1 edit is settled: `prepare_conditioning` returns `time_emb` unchanged, and `ClassConditioner`
wraps one zero-initialized `AdaLNBlock` per block, modulating each post-block feature map from the class
embedding (the full scaffold module is in the answer). The training objective is unchanged in shape — the
plain noise-MSE — only the network now sees the class through these adaptive-norm gates; sampling feeds
the fixed class index at every DDIM step.

Now reason about what this floor should do and where it should run out, because that sets up the next
rung. The adaptive-norm conditioner gives, per channel, one scale and one shift, computed from the label
and applied identically at *every* spatial location. For steering toward one of ten classes that is real
but blunt: the class can say "globally emphasize channel 37, suppress channel 12," and through the gate it
can do so with progressively more strength as training proceeds — but it cannot say "near this corner of
the image emphasize this, over there emphasize that," because the modulation has no spatial address and
never looks at the local content of the feature it modulates. It is a global, content-agnostic per-channel
knob, the same profile as the timestep. For CIFAR-10's ten coarse classes that should already produce a
genuinely class-conditional model with a sensible FID at every scale — the floor *works*, it is not
broken. But there are two structural reasons I expect it to be the *weakest* of the conditioning operators
I will try. First, I deliberately routed the class only through the *post-block* path while leaving the
timestep alone, so the class signal reaches the residual blocks' own AdaGN not at all — it only gets to
nudge the feature *after* each block, which is a strictly thinner channel than letting the class ride the
block-internal modulation the way the timestep does. Second, even granting that, the operator is
spatially uniform: it is the lowest-bandwidth of the affine, attention, and combined options, exactly the
mechanism that the strongest pixel models found insufficient on its own. So my falsifiable expectation is
that this adaptive-norm-only conditioner lands a real but comparatively *high* FID at all three scales —
the floor against which a method that *also* feeds the class through the block-internal AdaGN time path, or
that adds spatially content-dependent conditioning, should measurably improve. If a cheaper operator that
simply rides the class on the timestep embedding beats this, that tells me the post-block-only routing was
the bottleneck; if a content-dependent operator beats it, that tells me the spatial uniformity was. Either
way the diagnosis for step 2 is already pointed: the question is whether the next gain comes from a
*better route* for the same affine, or from a *richer* operator — and the cheapest thing to try first is
the better route.

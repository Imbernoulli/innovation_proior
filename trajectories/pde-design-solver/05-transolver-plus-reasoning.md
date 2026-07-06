Transolver did exactly what I bet whole-domain physics attention would do, and it did it on every metric
that the graph rungs left on the table. Let me set the deltas against GraphSAGE on paper, because they
are the confirmation that global correlation was the residual and they set the bar this finale must
clear. rho_d climbed to 0.9896 — finally past *both* the graph U-Net's 0.981 and GraphSAGE's 0.978,
which is the metric I said whole-domain attention should move most, confirming that a single
Physics-Attention layer relating the front-region state to the wake state directly is what orders designs
correctly. The drag-magnitude error fell to 0.0136 from GraphSAGE's 0.0193, down 30%. The field errors
beat GraphSAGE across both published benchmarks: Car pressure 0.0895 → 0.0809 (down 10%) and velocity
0.0330 → 0.0218 (down 34%); AirfRANS pressure 0.0458 → 0.0335 (down 27%) and velocity 0.0369 → 0.0156
(down 58%, better than a 2× improvement). The AirCraft probe I flagged as the uncertain one did diverge
upward — 0.684/0.411 against GraphSAGE's 0.636/0.380 — which is the overfitting-a-small-custom-set risk I
named, but the published benchmarks, where the comparison is meaningful, are a clean sweep. So the
strongest baseline is settled, and the verdict is that grouping points by *learned physical state* rather
than by location is the operator that closes the global-correlation gap. The only lever I explicitly left
open at the close of that rung was the slicing itself: *can the assignment be made sharper and the slices
more distinguishable than a single fixed-temperature softmax allows?* That is where I push now, and since
this is the endpoint of the ladder it has to be a genuine sharpening of the operator, not a restyling.

Let me diagnose Transolver's slicing precisely, because the move has to come out of a concrete weakness.
The slice assignment in Transolver is a softmax over the M slice axis with a *single learnable
temperature per head* — one scalar that scales the logits for every point in that head identically. Two
things about that are suboptimal once I stare at the geometry. First, the right sharpness is not uniform
across the domain. A point sitting squarely in the middle of a clean physical regime — deep in the wake,
or flat on the high-pressure nose — should commit hard to one slice, because it genuinely belongs to one
state; a point on a *boundary* between regimes (the separation line where flow detaches, a shock front,
the edge of the front region) is genuinely ambiguous and *should* be allowed to split across slices,
because it physically straddles them. A single per-head temperature cannot express "be decisive here, be
soft there"; it forces one sharpness on points whose ideal sharpness differs by where they sit in the
flow, so tuning it trades interior crispness against boundary honesty and can win neither. Second, even
with a well-tuned temperature, a plain softmax over only M=32 logits tends to leave the slices
*overlapping*: many points spread non-trivial mass across several slices, so the mass-normalized token of
a slice that shares half its points with three neighbors is a *blend* of states, not a clean one, and
attention among muddy tokens is attention among muddy states. The two weaknesses are coupled — a global
sharpness knob and a soft assignment that lets slices bleed into one another — and together they yield
tokens that are less *eidetic*, less sharply distinguishable, than the physics warrants. That the
velocity fields (0.0218 Car, 0.0156 AirfRANS) are already the sharper win and the pressure fields
(0.0809, 0.0335) the looser one tells me there is still field structure — the sharp separation and wake
boundaries — that blurred tokens are not resolving.

Let me quantify "blended" so the weakness is not just a word. A point's assignment is a distribution over
M=32 slices, and its overlap is measured by its entropy: a point that has committed to one slice has
entropy near 0 nats, while a point spread uniformly over all 32 slices has entropy `log 32 ≈ 3.47` nats —
the maximum. A plain softmax with a temperature tuned to keep the interior points crisp will, on the
boundary points and on any point whose logits happen to be close, leave assignments well up toward that
3.47-nat ceiling, meaning the point contributes its content to effectively many slices at once and every
one of those slices' tokens is contaminated by it. With N≈8000 points each smearing across several of
just 32 slices, the tokens overlap heavily — the mass-normalized token of a slice is an average over a
membership that is itself shared three or four ways — and that is the concrete sense in which attention
runs over muddy states. The eidetic target is to drive the typical per-point assignment entropy down
toward 0 *where the physics is clean* while leaving it high *only* where the physics is genuinely
ambiguous, which is precisely a per-point control problem that one global temperature cannot solve.

Before I commit to a fix I should walk the cheaper candidates for "sharpen the slicing," because if one
worked I would take it. The first is to simply *lower the fixed temperature globally* — make every head's
single scalar smaller so all points commit harder. That fails on the first weakness I just named: a
uniformly sharper softmax forces the ambiguous boundary points to commit as hard as the interior ones,
so the separation-line point that physically belongs half to "attached" and half to "separated" is
railroaded into one slice, and I would expect the boundary regions — exactly where the residual field
error lives — to get *worse*, not better. Uniform sharpness is the wrong axis. The second candidate is
to *raise M*, giving more slices so each can be narrower and overlap less. But I already reasoned at the
previous rung that M→N fragments the physics back into diluted point-attention, and there is a budget
cost too: more slices widen the slice-projection and the token attention, and I am running at the
budget-anchoring width, so I have no room to spend. More slices treat the symptom (overlap) by adding
capacity rather than by making the *assignment* itself decisive. The third candidate is the honest
target — a *hard* argmax assignment, each point to exactly one slice, which would give perfectly clean,
non-overlapping tokens. But argmax is non-differentiable: the slice-projection would get no gradient, the
assignment could never learn, and the model would be stuck with its random-init slices. So the real
problem is sharp and specific: I want near-categorical, non-overlapping assignments that are *still
differentiable*, and I want the sharpness to *vary per point*.

The fix that addresses both at once is to make the temperature **adaptive per point** and the assignment
**near-categorical without collapsing the gradient**. Take them in turn. For the temperature: instead of
one learned scalar per head, predict a temperature *for each point* from that point's own feature — a
small MLP that maps the per-head point feature through `dim_head → slice_num → 1` to a scalar, added to a
learnable per-head bias and floored at a small positive value (clamp at 0.01) so it never hits zero and
blows up the division. Now a point in a clean regime can learn a low temperature (sharp, decisive
commitment) while a boundary point can learn a high temperature (soft, hedged across the regimes it
straddles), and the model chooses this *locally* from the feature rather than globally. This is the exact
generalization of Transolver's single temperature, and I can verify the special case: if the
temperature-MLP weights go to zero, the predicted temperature is just the learnable per-head bias — a
per-head constant — so I recover Transolver's single-per-head-temperature softmax precisely. Transolver
is the constant-temperature point in this larger family, which is the check that I am strictly
generalizing and not replacing. For the decisiveness: replace the plain softmax over the slice logits
with a **Gumbel-softmax** — add Gumbel noise `−log(−log(u))` to the logits before the softmax. The
Gumbel-softmax is the continuous relaxation of *sampling* a hard one-hot categorical assignment: as its
temperature falls the sample concentrates toward one-hot, as it rises the sample spreads toward uniform,
and at every temperature it stays differentiable through the reparameterized noise. So with the adaptive
per-point temperature driving it, a clean-regime point (low temperature) draws a near-one-hot assignment
— the differentiable stand-in for the argmax I could not use directly — while a boundary point (high
temperature) draws a genuinely spread one. The noise also acts as a regularizer that breaks the lazy,
near-uniform assignment a plain softmax can settle into, because a point can no longer sit blandly at the
center of all 32 slices without the perturbed logits repeatedly pushing it toward a choice. Together
these make the slices *eidetic*: sharply distinguishable physical states, each token a clean
representative rather than a blend, because points commit decisively and the per-point temperature lets
the commitment vary with the local ambiguity. Attention among M clean tokens is then attention among
genuinely distinct states — the *same* O(M²) cost, but a sharper operator.

There is a second, independent gain hiding in this redesign, and I should take it while I am here because
it directly serves the task's parameter budget — which, being at the anchor width, is now binding.
Transolver's Physics-Attention uses a *two-stream* point projection: one linear map produces the feature
that *decides* the slice (`x_mid`), a separate linear map produces the *content* averaged into the token
(`fx_mid`). That decoupling is defensible, but it doubles the input-projection parameters, and once the
assignment is eidetic the content and assignment can share a single stream without much loss — a
sharply-committed point concentrates the right content into its one slice by the very act of committing,
so the separate content projection is buying less than it did when assignments were soft and blended. So
I collapse to a **single stream**: one projection `x_mid` serves both as the slice-decision feature *and*
as the content averaged into the token. Let me put the budget arithmetic on paper, because it is what
makes this admissible at equal width. Each input stream is a `Linear(dim, inner_dim) = Linear(256, 256)`,
so `256·256 + 256 ≈ 65.8k` parameters; dropping one of the two per attention layer, across 8 layers,
saves ~526k parameters. The adaptive-temperature MLP I added is `Linear(32,32) + Linear(32,1) ≈ 1.09k`
per attention layer, ~8.7k across 8 layers. So the single-stream collapse *more than pays* for the
temperature MLP — a net drop of roughly half a million parameters — which is exactly what lets the finale
run at the *same* `n_hidden=256, slice_num=32` as Transolver (for a fair head-to-head) while landing
comfortably *under* the Transolver-256 count, inside the task's 1.05×-Transolver-256 budget. That is the
30–70% footprint reduction the eidetic redesign is known for, and on this edit surface, where the cap is
defined against Transolver-256 itself, dropping a stream is what makes the upgrade admissible at all
rather than a nice-to-have.

One worry I should discharge before committing is whether collapsing to a single stream cripples the
content pathway — after all, `x_mid` now has to be both the assignment feature and the averaged content.
The reassurance is in where the content is actually *transformed*: the averaged token is not used raw; it
passes through the learned `to_q/to_k/to_v` maps and the token attention before being desliced, so the
model still has a dedicated, learnable content transformation — it simply lives *after* the slicing
rather than in a second input projection *before* it. What the two-stream bought was the freedom to
average a *different* linear view of the input into the token than the one used to assign it, and once the
assignment is near-one-hot that freedom is worth little: a point that deposits essentially all its mass
into one slice has already selected which token receives it, and the post-attention value maps recover any
needed reshaping of the content. So the single stream loses a projection whose marginal value the eidetic
assignment has itself made small — which is why the collapse is a genuine simplification rather than a
forced budget sacrifice.

Now I land this on the task's edit surface, and there is a structural subtlety. The task ships
`Physics_Attention_Irregular_Mesh` as a *read-only* module in `layers.Physics_Attention`; I cannot edit
it. But the edit surface lets me rewrite the entire `Custom.py` body (lines 1–64), so I define the
eidetic attention class **inline in Custom.py** — the same scaffold move Transolver used for its
`Transolver_block`, just one level deeper into the sublayer. The block and `Model` wrapper stay
byte-for-byte the canonical pre-norm Transolver structure (`fx = Attn(LN(fx)) + fx; fx = mlp(LN(fx)) +
fx`, last block carrying the read-out, `preprocess` lifting `fun_dim + space_dim → n_hidden`, the learned
`placeholder` bias, `geo` ignored), because the contribution is entirely in the attention sublayer, not
the skeleton. Inside the sublayer I must strip two things from the reference implementation that do not
apply here. The `torch.distributed.nn.all_reduce` calls sum slice statistics across GPUs in the
million-point multi-GPU setting; here it is one mesh on one device, batch size one, so there is nothing
to all-reduce — the slice mass `Σ_n w_ng` is already the complete sum over the single mesh. And the
gradient-checkpointing wrapper is a memory optimization, not part of the algorithm; the frozen loop
handles training and the meshes are small, so recomputation buys nothing. What remains is exactly the
algorithm: single-stream `x_mid` projection, a `proj_temperature` MLP plus a learnable bias clamped to a
small positive floor, Gumbel-softmax slice weights with that adaptive temperature, mass-normalized
tokens, scaled-dot-product attention among the M tokens, deslice through the same weights, output
projection. The slice projection keeps the **orthogonal initialization** — decorrelated slice directions
at init so the 32 slices specialize onto distinct feature axes rather than collapsing into duplicates —
inherited from Transolver and equally motivated here.

Let me trace the eidetic sublayer's shapes once, since a dimension check is the only verification
available before a run. With N=8000, n_hidden=256, heads=8 (dim_head=32), M=32: the single-stream
projection gives `x_mid` of shape `(B, 8, 8000, 32)`; the temperature MLP maps the trailing 32-dim
feature to a scalar, `(B, 8, 8000, 1)`, added to the `(1,8,1,1)` bias and clamped — a per-point,
per-head temperature; the slice projection maps the 32-dim feature to M=32 logits `(B, 8, 8000, 32)`, the
Gumbel-softmax over that last axis leaves a near-categorical partition of unity per point; the weighted
sum over the 8000 points contracts to tokens `(B, 8, 32, 32)`, the point axis replaced by the 32-slice
axis; scaled-dot-product attention among the 32 tokens returns `(B, 8, 32, 32)`; the deslice contracts
the 32 tokens back through the *same* `(B,8,8000,32)` weights to `(B,8,8000,32)`; heads recombine to
`(B,8000,256)` and the output projection returns it. Every contraction axis matches and N appears only in
the O(NM) encode and broadcast, never inside the attention — the same linear-in-N structure as
Transolver, now with the two extra tensors (per-point temperature, Gumbel noise) that make the assignment
eidetic, and one fewer input projection.

Let me make the bar this finale must clear explicit and falsifiable against Transolver's measured
numbers, since it carries no feedback of its own — it is the endpoint. The eidetic redesign targets
sharper, more distinguishable slices, which should most help where physical regimes have *sharp
boundaries* that a soft single-temperature softmax blurs. So on the published benchmarks I expect it to
**at least match, and aim to beat, Transolver's rho_d 0.9896 and c_d 0.0136**, and to **reduce the field
errors below Transolver's Car 0.0809/0.0218 and AirfRANS 0.0335/0.0156** — particularly the *velocity*
fields, where sharp separation and wake boundaries are exactly the ambiguous regions an adaptive
per-point temperature is built to handle, and where a boundary point that Transolver forced to commit can
now honestly hedge. The drag *ordering* is already near-saturated at 0.99, so the clearer win to look for
is the field relative-L2, where there is more headroom, and pressure specifically, since it is the looser
of Transolver's two fields. The strict requirement is that it not *regress* on the published benchmarks
while staying *under* the Transolver-256 parameter budget — and the single-stream collapse is what
guarantees the headroom, by the ~half-million-parameter arithmetic above. If the eidetic slicing genuinely
produces cleaner tokens, the field errors should fall and the parameter count should drop at the same
time; if instead it merely matches Transolver, the redesign bought efficiency (fewer parameters) without
accuracy, which is still admissible but weaker. The AirCraft probe I would again treat as task-internal
and not read as a verdict either way, given Transolver already diverged there to 0.684/0.411.

The causal chain, threaded from Transolver's result: Transolver swept the published benchmarks (rho_d
0.9896, c_d 0.0136 down 30% from GraphSAGE, Car 0.0809/0.0218, AirfRANS 0.0335/0.0156 with velocity down
58%), confirming that grouping points by learned state closes the global-correlation gap and leaving only
the *slicing's sharpness* as an open lever → diagnose Transolver's single per-head temperature as unable
to vary commitment by location and its plain softmax as leaving slices overlapping, so tokens are blended
rather than eidetic → reject the cheap fixes (a globally lower temperature over-commits the boundary
points; a larger M fragments and costs budget; a hard argmax is non-differentiable) → make the
temperature **adaptive per point** (a small `dim_head→slice_num→1` MLP plus a floored per-head bias, with
Transolver recovered as the zero-MLP special case) so clean-regime points commit hard and boundary points
hedge, and replace the slice softmax with a **Gumbel-softmax** that draws near-categorical yet
differentiable assignments and regularizes against the lazy uniform one → take the **single-stream**
projection collapse as a budget-critical efficiency (saving ~526k parameters against the ~8.7k the
temperature MLP adds) that keeps the finale under the 1.05×-Transolver-256 cap at equal width → land it
inline in `Custom.py` (the read-only Physics-Attention layer cannot be edited), stripping the reference's
multi-GPU all-reduce and gradient-checkpointing as inapplicable to one mesh on one device, at
**n_hidden=256, slice_num=32** → expecting it to match-or-beat Transolver's rho_d and c_d and to lower the
field errors (velocity most, then pressure), while using *fewer* parameters than the strongest baseline it
must surpass. The full scaffold module is in the answer.

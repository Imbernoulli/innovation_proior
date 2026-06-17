Transolver did exactly what I bet whole-domain physics attention would do, and it did it on every
metric that the graph rungs left on the table. rho_d climbed to 0.9896 — finally past *both* the graph
U-Net's 0.981 and GraphSAGE's 0.978, confirming that the residual after the graph trilogy really was
global correlation, and that a single Physics-Attention layer relating the front-region state to the
wake state directly is what orders designs correctly. The drag-magnitude error fell to 0.0136 (from
GraphSAGE's 0.0193), and the field errors beat GraphSAGE across both published benchmarks: Car pressure
0.0809 and velocity 0.0218 (GraphSAGE 0.0895/0.0330), AirfRANS pressure 0.0335 and velocity 0.0156
(GraphSAGE 0.0458/0.0369). The AirCraft probe I flagged as the uncertain one did diverge upward —
0.684/0.411, slightly worse than GraphSAGE's 0.636/0.380 — which is the overfitting-a-small-custom-set
risk I named, but the published benchmarks, where the comparison is meaningful, are a clean sweep. So
the strongest baseline is settled, and the verdict is that grouping points by *learned physical state*
rather than by location is the operator that closes the global-correlation gap. The only lever I
explicitly left open at the close of that rung was the slicing itself: *can the assignment be made
sharper and the slices more distinguishable than a single fixed-temperature softmax allows?* That is
where I push now.

Let me diagnose Transolver's slicing precisely, because the next move has to come out of a concrete
weakness, not a vague "make it better." The slice assignment in Transolver is a softmax over the M
slice axis with a *single learnable temperature per head* — one scalar that scales the logits for
every point in that head identically. Two things about that are suboptimal once I stare at the
geometry. First, the right sharpness is not uniform across the domain. A point sitting squarely in the
middle of a clean physical regime — deep in the wake, or flat on the high-pressure nose — should commit
hard to one slice; a point on a *boundary* between regimes (the separation line where flow detaches, a
shock front, the edge of the front region) is genuinely ambiguous and should be allowed to split
across slices. A single per-head temperature cannot express "be decisive here, be soft there"; it
forces one sharpness on points whose ideal sharpness differs by where they sit in the flow. Second,
even with a tuned temperature, a plain softmax over only M=32 logits tends to leave the slices
*overlapping* — many points spread non-trivial mass across several slices, so the tokens are not as
distinguishable as they could be, and attention among muddy tokens is attention among muddy states.
The mass-normalized token of a slice that shares half its points with three neighbors is a blend, not a
clean physical state. So the two weaknesses are coupled: a global sharpness knob and a soft assignment
that lets slices bleed into one another, yielding tokens that are less *eidetic* — less sharply
distinguishable — than the physics warrants.

The fix that addresses both at once is to make the temperature **adaptive per point** and to make the
assignment **more decisive** without collapsing the gradient. Take them in turn. For the temperature:
instead of one learned scalar per head, predict a temperature *for each point* from that point's own
feature — a small MLP that maps the per-head point feature to a positive scalar, added to a learnable
bias and floored at a small positive value so it never hits zero. Now a point in a clean regime can
learn a low temperature (sharp, decisive commitment to its slice) while a boundary point can learn a
high temperature (soft, hedged across the regimes it straddles), and the model chooses this locally
from the feature, not globally. This is the natural generalization of Transolver's single temperature:
Transolver is the special case where the predicted temperature is constant across points. For the
decisiveness: replace the plain softmax over the slice logits with a **Gumbel-softmax** — add Gumbel
noise to the logits before the softmax. The Gumbel-softmax is the continuous relaxation of sampling a
hard one-hot assignment; with the adaptive temperature controlling its sharpness, it pushes each
point's assignment toward a near-categorical choice (a point mostly *belongs* to one slice, the way a
hard argmax would assign it) while staying differentiable, so the slice-projection still gets a clean
gradient. The noise also acts as a regularizer that prevents the lazy near-uniform assignment a plain
softmax can settle into. Together these make the slices *eidetic*: sharply distinguishable physical
states, each token a clean representative rather than a blend, because points commit decisively and the
per-point temperature lets the commitment vary with the local ambiguity. The attention among M clean
tokens is then attention among genuinely distinct states — the same O(M²) cost, but a sharper
operator.

There is a second, independent gain hiding in this redesign that I should take while I am here, because
it directly serves the task's parameter budget. Transolver's Physics-Attention uses a *two-stream*
point projection: one linear map produces the feature that *decides* the slice (`x_mid`), a separate
linear map produces the *content* that gets averaged into the token (`fx_mid`). That decoupling is
defensible, but it doubles the input-projection parameters, and once the assignment is made eidetic the
content and assignment can share a single stream without much loss — the sharply-committed assignment
already concentrates the right content into each token. So I collapse to a **single stream**: one
projection `x_mid` serves both as the slice-decision feature *and* as the content averaged into the
token. That removes an entire `Linear(dim, inner_dim)` per attention layer. The reason this matters
*here specifically* is the budget: the harness rejects any model over 1.05× Transolver-256, and I want
to run this finale at the *same* `n_hidden=256, slice_num=32` to make the comparison fair — so I cannot
afford to *add* parameters. The adaptive-temperature MLP is small (`dim_head → slice_num → 1`), and
dropping the second input stream more than pays for it, so the finale lands *under* the Transolver
parameter count at equal width, comfortably inside the budget while being a strictly richer slicing
operator. That is the 30–70% footprint reduction the method is known for, and on this edit surface it
is what makes the upgrade admissible at all.

Now I have to land this on the task's edit surface faithfully, and there is a structural subtlety. The
task ships `Physics_Attention_Irregular_Mesh` as a *read-only* module in `layers.Physics_Attention`; I
cannot edit it. But the edit surface lets me rewrite the entire `Custom.py` body (lines 1–64), so I
define the eidetic attention class **inline in Custom.py** — the same scaffold move Transolver uses for
its `Transolver_block`, just one level deeper. The block and `Model` wrapper stay byte-for-byte the
canonical pre-norm Transolver structure (`fx = Attn(LN(fx)) + fx; fx = mlp(LN(fx)) + fx`, last block
carrying the read-out, `preprocess` lifting `fun_dim + space_dim → n_hidden`, the learned `placeholder`
bias, `geo` ignored), because the contribution is entirely in the attention sublayer, not the
skeleton. Inside the sublayer I must also strip two things from the reference implementation that do
not apply here: the `torch.distributed.nn.all_reduce` calls (those sum slice statistics across GPUs in
the million-scale multi-GPU setting; here it is one mesh on one device, batch size one, so there is
nothing to all-reduce) and the gradient-checkpointing wrapper (a memory optimization, not part of the
algorithm; the frozen loop handles training). What remains is exactly the algorithm: single-stream
`x_mid` projection, a `proj_temperature` MLP plus a learnable bias clamped to a small positive floor,
Gumbel-softmax slice weights with that adaptive temperature, mass-normalized tokens, scaled-dot-product
attention among the M tokens, deslice through the same weights, output projection. The slice projection
keeps the orthogonal initialization (decorrelated slice directions at init), inherited from Transolver
and equally motivated here.

Let me make the bar this finale must clear explicit and falsifiable against Transolver's measured
numbers, since it carries no feedback of its own. The thing the eidetic redesign targets — sharper,
more distinguishable slices — should most help where physical regimes have sharp boundaries that a soft
single-temperature softmax blurs. On the published benchmarks that means I expect it to **at least
match, and aim to beat, Transolver's rho_d 0.9896 and c_d 0.0136**, and to **reduce the field errors
below Transolver's Car 0.0809/0.0218 and AirfRANS 0.0335/0.0156** — particularly the velocity fields,
where sharp separation and wake boundaries are exactly the ambiguous regions an adaptive per-point
temperature is built to handle. The drag *ordering* (rho_d) is already near-saturated at 0.99, so the
clearer win to look for is the field relative-L2, where there is more room. The strict requirement is
that it not *regress* on the published benchmarks while staying *under* the Transolver-256 parameter
budget — and the single-stream collapse is what guarantees the budget headroom. If the eidetic slicing
genuinely produces cleaner tokens, the field errors should fall and the parameter count should drop at
the same time; if instead it merely matches Transolver, the redesign bought efficiency (fewer
parameters) without accuracy, which would still be admissible but weaker. The AirCraft probe I would
again treat as task-internal and not read as a verdict either way, given Transolver already diverged
there.

The causal chain, threaded from Transolver's result: Transolver swept the published benchmarks (rho_d
0.9896, c_d 0.0136, Car 0.0809/0.0218, AirfRANS 0.0335/0.0156), confirming that grouping points by
learned state closes the global-correlation gap and leaving only the *slicing's sharpness* as an open
lever → diagnose Transolver's single per-head temperature as unable to vary commitment by location and
its plain softmax as leaving slices overlapping, so tokens are blended rather than eidetic → make the
temperature **adaptive per point** (a small MLP per point, biased and floored) so clean-regime points
commit hard and boundary points hedge, and replace the slice softmax with a **Gumbel-softmax** that
pushes near-categorical, differentiable assignments → take the **single-stream** projection collapse as
a free, budget-critical efficiency that pays for the temperature MLP and keeps the finale under the
1.05× Transolver-256 cap at equal width → land it inline in `Custom.py` (the read-only Physics-Attention
layer cannot be edited), stripping the reference's multi-GPU all-reduce and checkpointing as
inapplicable, at **n_hidden=256, slice_num=32** → expecting it to match-or-beat Transolver's rho_d and
c_d and to lower the field errors (velocity most), while using *fewer* parameters than the strongest
baseline it must surpass. The full scaffold module is in the answer.

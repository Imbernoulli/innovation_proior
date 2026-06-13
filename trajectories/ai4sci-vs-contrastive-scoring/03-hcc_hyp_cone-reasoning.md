The HCC numbers confirm the ordering diagnosis and, in the same breath, expose the limit the ranking term
cannot reach. LIT-PCBA moved exactly where I predicted: AUROC climbed 0.576 → 0.628 and BEDROC 0.065 →
0.087, with EF@0.5% nearly doubling 7.27 → 11.15. So activity-aware ranking plus the sequence view did
attack the graded-actives failure — the relative gains *are* concentrated on the realistic benchmark,
just as the falsifiable claim said. DUD-E even firmed up its AUROC (0.895 → 0.920) and DEKOIS held
(0.892 → 0.922). But read LIT-PCBA again: 0.628 AUROC is still soft, and 0.087 BEDROC is still an order
of magnitude below the DUD-E number. The ranking term helped, and it stopped helping while the realistic
benchmark is still mostly unsolved. That is the tell I flagged as the alternative diagnosis at rung two:
if LIT-PCBA only partly moved, the residual problem is not ordering but the *geometry* — the Euclidean
space itself collapsing the cases that matter most. So now I have to stare at the geometry.

Where does Euclidean space fail, specifically? Activity cliffs. Two ligands that are almost the same
molecule — one substituent removed — that bind tens of fold differently. A structure encoder, doing its
job, maps near-identical molecules to near-identical vectors. In Euclidean space that is fatal, because
distance is linear: if $\|v_1-v_2\|$ is small in the encoder's native space it stays small in the
embedding, and any pocket-to-ligand score I read off is essentially the same for both. The ranking term I
just added cannot fix this — it asks the score to order the pair, but the two embeddings are so close
that there is no room to *place* the order without forcing structurally similar molecules to large
distance, which would fight the encoder's smoothness everywhere and wreck the genuinely-similar-and-
similar-affinity pairs too. And this is exactly LIT-PCBA's regime: realistic actives include
near-identical analogues, so the residual 0.628/0.087 is the model failing on precisely the pairs where
function diverges but structure does not. The ranking loss ran out of room because the *space* has no
room.

So what kind of space has room? Negative curvature. The defining fact about hyperbolic space is that
volume grows exponentially with radius instead of polynomially — that is why it embeds trees and
hierarchies with low distortion, there is exponentially more room near the boundary than a flat space
provides. And graded binding affinity *is* a hierarchy: within an assay ligands fan out from weak to
strong, and across the library there is a coarse-to-fine structure of "binds this family / binds this
target / binds this pocket specifically." If a curved space naturally encodes hierarchy, maybe it encodes
the affinity hierarchy, and — this is the bet — maybe the exponential growth gives me exactly the room to
separate cliffs cheaply, *without* uniformly stretching the metric.

Let me make that bet quantitative before committing, because "exponential room" is a slogan until I can
show a small structural difference becomes a large distance. Work on the Lorentz model: points on the
upper sheet of a hyperboloid with the Lorentzian inner product, geodesic distance
$d_L(x,y)=(1/\sqrt\kappa)\,\mathrm{arccosh}(-\kappa\langle x,y\rangle_L)$, and lift Euclidean encoder
outputs onto it with the exponential map at the origin. Take two ligands as tangent vectors at the origin
with nearly equal radial norm $r$ and a small angle $\theta\ll1$ between them. The hyperbolic law of
cosines gives, on a unit-curvature sheet, $\cosh d=\cosh^2 r-\sinh^2 r\cos\theta$; expand for small
$\theta$ with $\cos\theta\approx1-\theta^2/2$ and use $\cosh^2 r-\sinh^2 r=1$, so $\cosh d-1\approx
(\sinh^2 r/2)\theta^2$; invert with $\mathrm{arccosh}(1+\epsilon)=\sqrt{2\epsilon}$ to get
$d\approx\sinh r\cdot\theta$, and restoring curvature, $d_H\approx(\sinh r/\sqrt\kappa)\,\theta$. There
it is — the separation is $\theta$ *amplified by $\sinh r$*, which grows exponentially in $r$, where
Euclidean gives only $r\cdot\theta$, linear. So a cliff pair — tiny $\theta$, very different function — can
be separated if the geometry gives the stronger tier access to larger radial scale and tighter angular
control: the angular sliver is amplified at the radius where it matters, and I never distort the metric
uniformly. The bet has teeth. The design now reduces to: make the radial coordinate carry binding-strength
tier information, make angular position carry identity, and let cliffs separate through the product of the
two — exactly the room the HCC ranking term lacked.

How do I *control* radial depth and angular spread per ligand as a function of affinity? I need a
structured prior on the manifold, not "embed and hope." Entailment cones give it: attach to each point a
cone opening away from the origin, and the cone's half-aperture has a closed form that *shrinks* as the
point's norm grows — a point farther out projects a narrower cone. Read in my setting: a pocket pushed
deep toward the boundary is "more specific" and should admit only a tight set of ligand directions —
exactly the inductive bias I want for a specific binding pocket. The aperture transfers to the Lorentz
model as $\omega(x)=\arcsin(2K/(\sqrt\kappa\|x_{\text{space}}\|))$. But a single on/off cone is binary,
and I established I need *graded* tiers because affinity is graded. So I do the thing a plain cone does
not: turn it into a hierarchy of tiers indexed by affinity, using *both* the radial dimension (geodesic
pocket-ligand distance) and the angular dimension (the cone) as graded constraints.

Two measurements per ligand, and the argument order matters. The radial one is the geodesic distance
$d_{ij}$ from pocket $i$ to ligand $j$. The angular one is the first-argument angle
$\phi_{ij}=\mathrm{oxy\_angle}(\text{ligand}_j,\text{pocket}_i)$ in the hyperbolic triangle O-ligand-pocket,
and the aperture $\omega_i$ is attached to the *pocket*. The constraint the code imposes is
$\phi_{ij}\le\eta_{ij}\cdot\omega_i$ with the pocket supplying the aperture and the ligand-first angle the
measured lean — swapping the arguments would be a different loss, so I take the order the harness's lorentz
helpers compute. Now the tiers: bucket each ligand's pIC50 by the standard activity thresholds 5, 7, 9 (5
is the ~10 µM "active" cutoff, each step a decade), giving four buckets $b\in\{0,1,2,3\}$. Per ligand,
$r_k=r_0+b\,\Delta r$ and $\eta_k=\eta_0-b\,\Delta\eta$ with $r_0,\eta_0$ the weakest-tier base and
$\Delta r,\Delta\eta>0$. Stare at the signs because getting them backwards inverts the prior: a *stronger*
binder (larger $b$) gets a *larger* radial cap — a one-sided hinge permitting it to occupy the larger
radial scales where the $\sinh r$ amplification can act — and a *smaller* angular tolerance, because a
strong, specific binding event should align more decisively with the pocket's admissible direction than a
weak one. Strong = larger radial cap, tighter cone; weak = smaller cap, wider cone. The two knobs move in
opposite directions with affinity, spreading tiers by both distance and angular selectivity — exactly the
two-axis separation the cliff calculation said I needed, since $\sinh r\cdot\theta$ uses both.

The cone losses are one-sided hinges — penalize only violations, never pull a satisfied ligand:
$L_{\text{rad}}=(1/\sqrt N)\sum\max(d_{ij}-r_{ij},0)$, $L_{\text{ang}}=(1/\sqrt N)\sum\max(\phi_{ij}-
\eta_{ij}\omega_i,0)$, combined as $\lambda_{\text{rad}}L_{\text{rad}}+\lambda_{\text{ang}}L_{\text{ang}}$
with both weights 0.5 (two halves of the same cone, no reason to prefer one). The $1/\sqrt N$ is the same
assay-size discipline I have used since rung one — sub-linear so big assays count more but do not drown the
many small ones. Two regularizers fall out of what can go wrong with the cone. The angular hinge is zero
the instant $\phi\le\eta\omega$, so the optimizer has no pressure to do better than touch the boundary and
could collapse angles toward the axis trivially; a margin $m=0.15$ rad — $R_{\text{ang}}=(1/\sqrt N)\sum
\max(\phi-\eta\omega+m,0)$ — keeps pushing until ligands are decisively *inside* the cone. And because the
metric is dominated by the very top of the list, a heterogeneity term weights threshold-selected entries by
distance rank with $w_j=\exp(-\beta(\text{rank}_j-1)/L_i)$ at $\beta=80.5$ — the *same* focus parameter as
BEDROC, so I shape training to match the evaluation's early-enrichment emphasis. Both regularizers are
auxiliary, weight 0.10 each.

The cone hierarchy is the new part, but it rides on top of a retrieval objective that still has to find
binders at all — so I keep HCC's contrastive-plus-ranking core from rung two, applied to the hyperbolic
embeddings, and I am careful about the similarity I feed the softmax. Geodesic distance is the "correct"
hyperbolic similarity, but at inference I need a plain dot product so retrieval stays a matmul over a cached
matrix — non-negotiable, it is the whole reason screening is feasible. So I score with the inner product of
the spatial components both inside the softmax logits and at inference, and let the cone losses do the
geometric shaping. And here the bookkeeping detail I deliberately *omitted* at rung two becomes
load-bearing: the exp map returns only the space components, and the implementation treats the projector
output as $[\text{lead}, \text{space}\ldots]$, dropping index 0 with `emb[:, 1:]` before the similarity, so
the manifold coordinates stay aligned. At rung two I noted there was no reserved coordinate to drop because
the space was flat; on the manifold the `[:, 1:]` slice is exactly the convention that keeps the
contrastive/cone math consistent. At *inference*, though, the score uses the *full* embedding dot product
(`pocket_reps @ mol_reps.T`, max over the target's pockets, plus the sequence contribution), because that
is the cached-retrieval matmul as the embeddings are produced — the geometry was paid for at training time.

The projection heads and the parameters the geometry needs come with initialization traps I have to defuse.
Each backbone feature goes through the same NonLinearHead into 128-d, but then the exp map scales by
$\sinh$, and CLIP-style init makes the Euclidean output have norm $\approx\sqrt n$, so $\sinh(\sqrt\kappa
\sqrt{128})$ is astronomical and training diverges immediately. The fix: a learnable per-tower scale
$\alpha$ initialized to $1/\sqrt n=128^{-1/2}$ so the scaled embedding has expected unit norm at init,
learned in log space, clamped so $\exp(\alpha)\le1$ — the scale can shrink but never blow the exp map back
up. So each projection is $u=\text{head}(\text{feat})\cdot\exp(\alpha)$, then $h=\exp\_map0(u,\kappa)$. The
curvature $\kappa$ is itself learnable (init $\log 1$), clamped to $[\log 0.1,\log 10]$ so it can neither
collapse to Euclidean nor detonate. The contrastive softmax keeps the detached $\log 13$ inverse
temperature. The sequence pathway runs the identical contrastive-plus-ranking helper as a second query
view, but I apply the cone supervision *only* to the pocket branch — the pocket carries the structural,
geometrically meaningful signal, and over-constraining the sequence branch geometrically would fight its
role as an auxiliary view. So the total is $\alpha_{\text{poc}}\text{loss}_{\text{poc}}+\alpha_{\text{prot}}
\text{loss}_{\text{seq}}+\gamma_{\text{cone}}L_{\text{cone}}+\lambda_{\text{het}}R_{\text{het}}+
\lambda_{\text{ang}}R_{\text{ang}}$ with $\gamma_{\text{cone}}=0.1$, down-weighted because the cone is a
structural prior shaping the space, not the primary retrieval signal — if it dominated, the model would
satisfy the geometry at the expense of actually finding binders. (The full scaffold module is in the
answer.)

So the delta from HCC is the geometry: keep the contrastive-plus-ranking core verbatim, lift every
embedding onto a learnable-curvature Lorentz hyperboloid through a clamped, scaled exp map, and add an
affinity-tiered radial-plus-angular cone hierarchy (pocket branch only) with a margin and a
BEDROC-matched heterogeneity regularizer. Reading HCC's numbers, the falsifiable claims are sharp and they
are specifically about the realistic benchmark, because that is where cliffs live. LIT-PCBA's AUROC and
BEDROC should rise *again* past HCC's 0.628 / 0.087 — if the $\sinh r$ room is real, the model should now
separate the near-identical-but-different-affinity pairs that the Euclidean ranking term could not place,
and DUD-E (already near its ceiling at 0.920 AUROC) should hold or edge up on the early-enrichment metrics
where the cone's head-of-list shaping helps. The honest risk is that the cone is a strong prior on top of
already-strong features: it could trade a little on a benchmark whose decoys are easy (DEKOIS, where the
extra geometric constraint is unnecessary) for a gain where cliffs dominate, so I would not be surprised to
see DEKOIS hold roughly flat rather than climb. The claim that decides whether the geometry was worth it is
DUD-E's BEDROC/EF and LIT-PCBA's AUROC moving up together — separation where it was already good, plus the
cliff room the flat space never had.

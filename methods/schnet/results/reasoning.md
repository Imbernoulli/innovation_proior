Let me start from what I actually need to predict and what physics will not let me get away with. I have a molecule: a set of atoms with nuclear charges Z_1,…,Z_n and positions r_1,…,r_n in R^3, and I want one model that gives me the total energy E(r_1,…,r_n) and, from the very same model, the forces on every atom. The forces aren't a separate target I get to fit freely — they're F_i = -∂E/∂r_i, the negative gradient of the energy. That single fact is going to discipline everything. If I instead trained a network to output a force vector per atom directly, decoupled from any scalar energy, nothing would stop the resulting vector field from having nonzero curl, and a force field with curl is not conservative: I could walk the atoms around a closed loop in configuration space and come back with more energy than I started, which is nonsense. So I want a scalar energy model and I want forces to be its gradient by construction. Then curl-freeness is automatic — the gradient of a scalar always has zero curl — and energy conservation comes for free.

Now what does the energy have to respect? It's a scalar physical observable, so rotating the whole molecule, translating it, or relabeling which atom I call atom 1 cannot change E. Rotation, translation, permutation invariance. And the force, being a gradient of an invariant scalar, will automatically be rotationally *equivariant* — rotate the molecule and each force vector rotates with it — which is exactly the physical behavior I want, and I get it for free the moment E is invariant. So the real design target collapses to: build a permutation- and rotation-invariant scalar function of the geometry, and let autodiff hand me the forces.

There's a third constraint that's easy to underrate and turns out to be the one that kills most of the obvious approaches: smoothness. The potential-energy surface and its derivatives have to be smooth in the atom positions. Geometry optimization needs a continuous gradient. And training on forces needs more than that — I'm going to put the force, which is already ∂E/∂r, inside a loss and descend it, so I need to differentiate the force, which means E has to be *twice* differentiable. Any place where E(R) has a kink or a jump becomes a spike or a hole in the force, and force training falls apart there. Hold onto this; it's the blade I'll keep cutting candidate designs with.

So how do people build representations of molecules today, and where does each one stall against these three constraints? The most physically careful camp writes the energy as a sum over atoms, E = Σ_i E_i, where each atomic contribution E_i depends only on the local environment of atom i — its neighbors within some cutoff radius r_cut. To feed each atom's little network, they hand-engineer a fixed vector of descriptors of that environment: radial functions of neighbor distances, angular functions of neighbor triplets, the atom-centered symmetry functions of Behler and Parrinello. These are genuinely good: they're invariant by construction, they're smooth (they even use a cosine cutoff so neighbors fade in and out without a jump — I'll come back to that, it's a lovely trick), they're energy-conserving when you take the gradient, and because each atom has a bounded number of neighbors inside r_cut the cost scales linearly with system size. The thing that bothers me is that the descriptors are *hand-built and frozen*. Somebody has to pick the radial widths, the angular resolutions, the cutoff, and a set tuned for one chemistry isn't automatically right for another. The representation isn't learned from the data; it's prescribed. I'd like the network to discover its own environment descriptors.

So learn the representation. The deep-tensor-network line does exactly that: start each atom from a learnable type-specific embedding c_i, and refine it over a few passes by accumulating messages from every other atom, c_i^{t+1} = c_i^t + Σ_{j≠i} v_ij, where the message v_ij couples the neighbor's features with the interatomic distance (Gaussian-expanded) through a parameter tensor — a bilinear coupling, kept affordable by a low-rank factorization. This is invariant, it's learned, it reaches good accuracy. But stare at how distance enters: through a factorized bilinear tensor that mixes the distance-expansion coefficients with the feature vector. It works, but it's an opaque coupling — there's no spatial-filter reading of it, no "this is a convolution" structure I can lean on or visualize, and in practice it leaves accuracy on the table and shares its interaction parameters across passes, which caps expressiveness. I want something with cleaner structure where the role of geometry is transparent.

The message-passing framework is the cleanest statement of the learned-representation idea: every graph net is m_v = Σ_{w∈N(v)} M(h_v, h_w, e_vw), then h_v ← U(h_v, m_v), then a readout over all nodes. The strongest molecular instance generates, from the edge features e_vw, a full F×F matrix A(e_vw) and sends the message A(e_vw)·h_w — an edge network — with a GRU update and a fancy set2set readout, and it takes the QM9 property benchmark. Beautiful framework. But here's where the smoothness blade comes down: its edge features include one-hot bond types — single, double, triple. The moment a bond stretches and the model's notion of "what kind of bond is this" flips, the one-hot input jumps, and the predicted energy jumps with it. The potential-energy surface is discontinuous. So this thing, for all its accuracy on relaxed molecules, cannot give me a smooth PES, cannot be trained on forces, cannot do molecular dynamics. It lives entirely in the equilibrium world where forces are zero. That's the camp I have to leave. And separately, the kernel approach in the gradient domain fits the force field directly with a kernel built so the force is the gradient of a scalar — energy-conserving by construction, stunningly data-efficient — but its Gram matrix grows quadratically in both the number of atoms and the number of training examples, so it simply doesn't scale to large data or to spanning many compositions. One model per molecule, and a wall at fifty thousand examples.

So I've got the tension sharp now. The descriptor methods are smooth, physical, scalable, but not learned. The learned graph nets are scalable and learned but tie their geometry to discrete, jumpy features that wreck the PES. I want the learned, scalable side *and* the smooth, physical side at once. Where exactly does the discreteness sneak in? It's the grid, or its molecular cousin, the bond-type bin. A normal convolutional layer has a filter W[Δ] indexed by integer offsets Δ — pixels, audio samples, evenly spaced. Atoms are not on a grid; they sit at arbitrary real positions. If I voxelize space to force them onto a grid, then as an atom slides across a voxel boundary its contribution snaps from one filter tap to the next: discontinuity, again. The bond-type one-hot is the same disease in a different coordinate. The root cause is that the filter is a lookup table over a *discrete* index.

What if the filter weren't a table indexed by a discrete offset, but a *function* of the continuous offset? Replace the filter tensor W[Δ] by a filter-generating function W^l: R^D → R^F that maps a relative position to a vector of filter values. Then the convolution at atom i becomes a sum over the other atoms,

  x_i^{l+1} = (X^l * W^l)_i = Σ_j x_j^l ∘ W^l(r_i - r_j),

where I'm letting W act on the actual continuous displacement r_i − r_j. There's no grid anymore: an atom at any position r_j contributes through W evaluated at its exact offset, and as it moves, W(r_i − r_j) moves with it continuously — no snapping, no bins. This is a continuous-filter convolution. The idea of generating a filter from a network isn't wholly new — dynamic filter networks already have a small net produce convolution weights conditioned on the input — but there the weights still land on a grid; here I'm cutting the grid loose entirely and letting the filter be defined at arbitrary positions and for an arbitrary number of neighbors. And I model W^l itself as a small neural network, because I want the filter learned, not prescribed.

Now I have to be careful about the "∘" and the cost. If W(r_i − r_j) were a full F×F matrix applied to x_j, every cfconv would cost F^2 per edge and I'd be back to the heavy edge-network message. I don't think I need the full matrix. Let me make the filter act *feature-wise*: W^l(r_i − r_j) is a vector in R^F and I multiply it elementwise with x_j^l, channel by channel. That's the depthwise-separable trick — do the spatial part per channel, cheaply, and handle cross-feature mixing separately with ordinary dense layers applied per atom. So the cfconv carries the geometry, and plain atom-wise layers recombine the channels. Linear in F, not quadratic, and the division of labor is clean: geometry in the filter, feature mixing in the dense layers.

But W^l(r_i − r_j) as written takes the raw displacement vector, and a raw vector is not rotation-invariant — rotate the molecule and r_i − r_j rotates, so the filter output would rotate, so E would not be invariant. That breaks constraint one. The fix is forced: feed the filter network only the *distance* d_ij = ‖r_i − r_j‖, which is invariant to rotation and translation. So the filter is a function of a single scalar per edge, W(d_ij) ∈ R^F, the message is x_j ∘ W(d_ij), and because the only geometric input is an invariant scalar, the whole energy is invariant and the forces fall out equivariant.

Before I trust the cost claim, let me actually trace the message-passing arithmetic on a tiny case, because the whole point of going feature-wise was to dodge the F^2 edge-network cost and I want to see that it holds. Three atoms, F = 4 channels, two directed edges feeding atom 0 from atoms 1 and 2. With projected features x_1 = (0,1,0,0), x_2 = (0,0,1,0) and per-edge filters W_{1→0} = (0.5,0.5,0.5,0.5), W_{2→0} = (0.2,0.2,0.2,0.2), the message into atom 0 is the elementwise product summed over neighbors, x_1 ∘ W_{1→0} + x_2 ∘ W_{2→0} = (0, 0.5, 0.2, 0). Each edge cost me F = 4 multiplies, not F^2 = 16 — and stacking more neighbors just adds more length-F vectors, so the per-edge work stays O(F). That's the depthwise-separable saving made concrete: had I used a full F×F matrix A(d_ij) instead of the elementwise vector, every edge would have cost the quadratic 16. So elementwise W is the diagonal special case of the edge-network message A(e)·h_w, with the discrete edge feature swapped for a continuous distance — same family, but linear in F and smooth in geometry. Good — that's what buys me both smoothness and cheapness in one move.

Let me sanity-check that this actually trains before I get attached to it. The filter network takes the scalar d_ij and must produce F filter values, one per channel. At initialization a neural network is close to linear, so each of the F output channels is approximately a linear function of the single input d_ij — which means all F channels are nearly the same linear ramp in d, just scaled. The filters are almost identical across channels: highly correlated, carrying essentially one degree of freedom instead of F. I've seen what that does — the optimization sits on a plateau at the start because there's no diversity among the filters to exploit, and it's hard to climb off. Wall. Feeding a bare scalar into the filter net is a bad idea precisely because one scalar through a near-linear net gives near-identical outputs.

How do I decorrelate the channels at initialization? I need to lift the single scalar d_ij into a representation where different channels naturally see different things. Pass the distance through a bank of localized basis functions before the dense layers — radial basis functions, Gaussians centered at a grid of distances,

  e_k(r_i − r_j) = exp( -γ (d_ij − μ_k)^2 ),

with centers μ_k spread on a uniform grid out to the cutoff. The intuition is that a given distance should light up the few Gaussians whose centers are near it and leave the rest near zero, so the input to the filter net is a sparse, localized bump rather than a single magnitude, and different output channels of the filter net can latch onto different centers — different distance ranges — so even a near-linear net at init produces *diverse* filters. But I should check the width, because if the Gaussians are too narrow a distance falling between two centers lights up nothing (the expansion becomes nearly all-zeros, no signal), and if too wide every center responds the same and I'm back to correlated channels. Lay the centers at offset = linspace(0, cutoff, K) with spacing Δ. The width γ = 1/(2Δ^2) is the same as standard deviation σ = Δ — one grid spacing. Let me test that this is the right balance with Δ = 1: at the midpoint between two adjacent centers, each of the two Gaussians evaluates to exp(−0.5·(0.5/1)^2) = 0.882, so a distance landing exactly between centers is still strongly registered by both — no dead zone. And it isn't washed out either: a distance at d = 0.3 with unit-spaced centers gives values (0.07, 0.43, 0.96, 0.78, 0.24, 0.03) on centers (−2,−1,0,1,2,3) — about five centers carry meaningful signal and the rest are negligible. So a single scalar becomes a localized five-ish-component pattern, which is exactly the decorrelation I wanted: distinct channels see distinct distance bands. That width is the right one. The plateau should then go away and training start moving — I'd want to confirm the plateau-removal empirically, but the channel-diversity mechanism is now concrete rather than hoped-for. And this expansion gives me two interpretable knobs for free: the number of centers is the resolution of the filter (more Gaussians, finer distance discrimination), and the range the centers span is the filter "size" — the analog of how big a patch a grid filter covers. I want the centers to span the distances that actually occur, so a grid from 0 out to the cutoff at a fine spacing (on the order of a tenth of an Ångström). So the filter network is: distance → Gaussian expansion → two dense layers → the F filter values. Crucially this expansion is also smooth in d (Gaussians are C^∞), so I haven't reintroduced any discontinuity; the smoothness blade is still satisfied.

Now the nonlinearity inside all these networks. ReLU is the default, but think about what I'm going to do with this model: differentiate it twice. The energy is the network; the force is its first derivative; the force loss is descended, taking a second derivative. ReLU has a corner — its first derivative is a step, its second derivative is a delta at zero. Put that anywhere in the energy network and the force gets a discontinuity and the second derivative blows up; force training breaks exactly at the kinks. I need an activation that's smooth to all orders. Softplus, ln(1 + e^x), is the C^∞ cousin of ReLU — it bends instead of kinking, and it has derivatives of every order. That satisfies the twice-differentiable requirement everywhere. One small adjustment: softplus(0) = ln(1 + 1) = ln 2 ≈ 0.693 ≠ 0, so a zero pre-activation doesn't map to zero, which biases the activations and hurts convergence. Shift it down by ln 2:

  ssp(x) = softplus(x) − ln 2 = ln(1 + e^x) − ln 2 = ln( (1 + e^x)/2 ) = ln( 0.5 e^x + 0.5 ).

Let me verify that last rewrite is actually an identity and not just algebra I want to be true, because I'll implement whichever form is cheaper and they had better agree. At x = 0: ssp(0) = ln 2 − ln 2 = 0, and ln(0.5·1 + 0.5) = ln 1 = 0 — both zero, as designed. At x = 2: softplus(2) − ln 2 = ln(1 + e^2) − ln 2 = 1.4338, and ln(0.5·e^2 + 0.5) = ln(4.1945) = 1.4338. At x = −0.5: ln(1 + e^{−0.5}) − ln 2 = −0.2191, and ln(0.5·e^{−0.5} + 0.5) = ln(0.8033) = −0.2191. They coincide everywhere I check, so the closed form is exact and either is fine to code. With this, ssp(0) = 0, the activations stay centered, and I keep the infinite differentiability. I'll use this shifted softplus throughout — in the filter net and in every atom-wise layer — because every one of those sits inside the energy that I'm going to differentiate twice.

Let me assemble the atom's life through the network. Each atom starts from a learnable embedding that depends only on its type, x_i^0 = a_{Z_i} — a vector per element, optimized during training; the network discovers its own per-element starting representation rather than me handing it a Coulomb descriptor. Then I interleave two kinds of layers. Atom-wise layers are ordinary dense layers applied independently and identically to every atom, x_i ← W x_i + b; because the weights are shared across atoms, the model is automatically permutation-equivariant and scales to any molecule size, and these layers do the cross-feature recombination that the feature-wise cfconv deliberately doesn't. And the cfconv layer does the geometry: gather each atom's neighbors, generate the distance filter, multiply feature-wise, sum over neighbors.

How do I stack these into something deep without it being a pain to train? Wrap each geometric update in a residual connection — x_i^{l+1} = x_i^l + v_i^l — so the block only has to learn a correction to the current representation, and a deep stack stays trainable. Inside the residual I want enough capacity to mix features both before and after the spatial mixing, so the block is: an atom-wise layer to set up the features, then the cfconv to let neighbors talk through the distance filter, then an atom-wise layer, a shifted-softplus nonlinearity, and one more atom-wise layer — and that whole thing is the residual v_i^l added back to x_i^l. One more choice: should the interaction blocks share weights across the stack, the way the recurrent message-passing and tensor nets do? I'll *not* share them. Each interaction block gets its own parameters. Sharing would tie every refinement pass to the same transformation; letting them differ lets earlier blocks build short-range structure and later blocks build on it, which is more expressive — and these repeated, distinct local updates are what let the network assemble genuinely many-body representations out of strictly pairwise filters, the same way stacking local convolutions builds large receptive fields in image nets.

I should pin down why stacking pairwise interactions gives many-body terms, because it's the conceptual payoff. One cfconv lets atom i feel each neighbor j individually — that's pairwise. But after one block, x_j has already absorbed information about j's own neighbors k. So in the next block, when i pulls in x_j, it's implicitly pulling in something that knows about k: i feels the (i,j,k) triple. Three blocks and i has a representation reflecting its radial environment several hops out, all while every individual filter only ever looked at a single invariant distance. The depth manufactures the many-body character; the invariance is never spent because each filter is radial.

Now the cutoff. I keep saying "neighbors within r_cut," and the cheap way to do that is a hard rule: include j in the sum iff d_ij < r_cut. But run the smoothness blade across it. During molecular dynamics an atom drifts in and out across the r_cut boundary. With a hard cutoff, the instant d_ij crosses r_cut, that neighbor's contribution snaps from whatever finite value W(d_ij) gave to exactly zero. The energy has a jump; the force has a delta. I've reintroduced the very discontinuity I built continuous filters to avoid, just at the boundary instead of inside. The descriptor methods already solved this — multiply each neighbor's contribution by a cosine cutoff,

  f_cut(d) = 0.5 · [ 1 + cos(π d / r_cut) ]   for d < r_cut,   0 for d ≥ r_cut.

I need to check that this actually glues to the outside-zero continuously to first order, otherwise it's a discontinuity in disguise. Value at the boundary: at d = r_cut, cos(π) = −1, so f_cut = 0.5·(1 − 1) = 0 — matches the zero just outside, so f_cut is continuous. Now the slope. Differentiating, f_cut'(d) = 0.5·(−sin(π d/r_cut))·(π/r_cut) = −(π/2r_cut)·sin(π d/r_cut), and at d = r_cut this is −(π/2r_cut)·sin(π) = 0 — matches the zero slope outside, so f_cut is C^1 across the join. Let me double-check the slope numerically rather than trust sin(π) = 0 by inspection: taking r_cut = 10 and a finite difference of f_cut just inside the boundary gives a slope of about −7·10^{−8}, i.e. numerically vanishing as d → r_cut. So both the value and the first derivative go smoothly to zero at the boundary — the neighbor fades out instead of vanishing. Fold this into the filter: the effective filter is the generated W(d) times f_cut(d), so W_eff = filter_net(e(d)) · f_cut(d). The energy stays C^1 across the boundary, the force stays continuous, and I still get the linear-cost locality of a bounded neighborhood. This is the same family as Behler's f_cut, repurposed to make the *learned* filter behave at its edge.

So now I can write the cfconv concretely as a message-passing step, since "gather neighbors, multiply by a filter, sum" is exactly a message function with an add-aggregation. For a directed edge (j → i) carrying the expanded distance e(d_ij): the message is x_j (after a feature-projection) times the cutoff-modulated filter, and I sum messages into i. Wrapping the feature projections around it: project x into the filter's channel count with one linear map (no bias — the filter supplies the scale), multiply feature-wise by W_eff = filter_net(e(d)) · f_cut(d), aggregate by summation over neighbors, then project back up to the hidden width. That projection-pair plus the elementwise filter multiply is the whole geometric core.

I also need to think about training the forces and the loss, because that's why I went to all this trouble to be twice differentiable. The force prediction is just the analytic gradient of the energy network, F̂_i = −∂Ê/∂r_i, which autodiff gives me by backpropagating the scalar energy to the input positions. Because Ê is rotation-invariant, this gradient is rotation-equivariant, and because it's literally a gradient it's curl-free and energy-conserving — all three physics constraints discharged by construction. To train on both energies and forces I want a loss that weights them sensibly:

  ℓ(Ê, (E, F_1,…,F_n)) = ρ ‖E − Ê‖^2 + (1/n) Σ_i ‖ F_i − ( −∂Ê/∂r_i ) ‖^2.

Energies and forces live on very different scales (one number per molecule versus 3n force components with their own magnitude), so the trade-off ρ has to down-weight the energy term; ρ ≈ 0.01 balances them in practice. The price of force training is compute: to get the forces I already do a forward and a backward pass through the energy model, so the force model is effectively twice as deep, and the force loss adds another backward on top — worth it, because tying energies and forces through one model is what makes it generalize off equilibrium from few samples.

Let me also choose the readout and the optimization. The energy is *extensive* — it scales with the number of atoms — so after a small atom-wise output network maps each atom's final representation to a scalar (or a small vector) contribution, I sum over atoms; an intensive property would average instead. Output network: an atom-wise layer down to a smaller width, shifted softplus, then a linear map to the output. For optimization, Adam with an initial learning rate of 1e-3 and exponential decay, small minibatches of molecules, and — because the last iterates of a stochastic optimizer rattle around — an exponential moving average over the weights for the model used at test time.

Now step back, because the same encoder is exactly what I'd want for a different problem: representing a protein from its alpha-carbon coordinates. A protein is a graph of residues at 3D positions, each carrying a feature vector, and I want per-residue and per-protein embeddings invariant to how the protein is oriented in space. Everything transfers. The geometry is still summarized by interatomic — here inter-residue — distances; the continuous filter is still rotation-invariant for the same reason; the residual stack of interaction blocks still builds multi-scale structure out of local radial filters; and I pool to get a per-graph vector. Two adjustments only, both forced by the data rather than the method: the input isn't a single atom type, it's a per-node feature vector, so the type-embedding becomes a lazy linear map from those input features into the hidden width; and the graph is already supplied as a spatial neighbor graph over residues, so I compute distances on those edges rather than asking the energy model to build a radius graph over atoms. The geometric core — distance → Gaussian expansion → filter net → cosine-cut, feature-wise filter, summed messages, residual blocks — is untouched. There are no forces to train here; the landing artifact is the encoder itself, producing node and graph embeddings that a downstream head consumes.

Let me write the geometric core as the pieces fall: the Gaussian distance expansion, the shifted softplus, the continuous-filter convolution with its cosine cutoff, the interaction block, and the encoder that stacks them.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class GaussianSmearing(nn.Module):
    """Lift a scalar distance d into a bank of localized RBFs e_k(d) =
    exp(-(d - mu_k)^2 / (2*Delta^2)), centers mu_k on a uniform grid 0..cutoff.
    Decorrelates the filter network's inputs so the F output channels start
    diverse instead of all near-identical (no early-training plateau)."""
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)   # centers mu_k
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2  # -1/(2*Delta^2) = -gamma
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)     # d - mu_k
        return torch.exp(self.coeff * dist.pow(2))            # exp(-gamma (d-mu_k)^2)


class ShiftedSoftplus(nn.Module):
    """ssp(x) = softplus(x) - ln 2 = ln(0.5 e^x + 0.5). C-infinity everywhere
    (needed so forces and their derivatives exist) and ssp(0) = 0 (centering)."""
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution: message j->i is x_j (projected) times the
    distance filter W(d_ij) = filter_net(e(d_ij)) * f_cut(d_ij), summed over j.
    Feature-wise (elementwise) so it costs O(num_filters) per edge, not O(F^2);
    cross-feature mixing is left to the atom-wise linear maps around it."""
    def __init__(self, in_channels, out_channels, num_filters, filter_net, cutoff):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)  # project into filter channels
        self.lin2 = nn.Linear(num_filters, out_channels)             # project back up
        self.filter_net = filter_net
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        # cosine cutoff: smoothly -> 0 (with zero slope) at d = cutoff, so a
        # neighbor crossing the boundary fades out without a jump in E or F.
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.filter_net(edge_attr) * C.view(-1, 1)              # filter values per edge
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)                    # sum_j x_j (*) W
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W                                              # feature-wise multiply


class InteractionBlock(nn.Module):
    """Residual interaction: filter net (RBF -> ssp -> dense) feeds the cfconv,
    then ssp and an atom-wise dense layer. Caller adds the result as a residual."""
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.filter_net, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class ProteinEncoder(nn.Module):
    """Continuous-filter encoder over alpha-carbon coordinates. Distances are
    rotation/translation-invariant, so node and graph embeddings are invariant
    by construction. Same geometric core as the molecular energy model; the only
    changes are a lazy linear embedding over input node features and a provided
    spatial edge_index."""
    def __init__(self, hidden_channels=128, out_dim=1, num_layers=6,
                 num_filters=128, num_gaussians=50, cutoff=10.0, readout="add"):
        super().__init__()
        self.cutoff = cutoff
        self.readout = readout

        self.embedding = nn.LazyLinear(hidden_channels)               # node features -> hidden
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_layers)
        ])
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.LazyLinear(out_dim)

    def forward(self, batch):
        edge_index = batch.edge_index
        u, v = edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)     # invariant distances
        edge_attr = self.distance_expansion(edge_weight)             # Gaussian-expanded
        h = self.embedding(batch.x)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)  # residual stack
        h = self.lin1(h)
        h = self.act(h)
        node_emb = self.lin2(h)
        graph_emb = scatter(node_emb, batch.batch, dim=0, reduce=self.readout)
        return {"node_embedding": node_emb, "graph_embedding": graph_emb}
```

Tracing the causal chain: I needed an invariant scalar energy whose gradient gives curl-free, energy-conserving, equivariant forces, with a potential-energy surface smooth enough to differentiate twice. The descriptor methods were smooth and physical but their environment features were hand-built and frozen; the learned graph nets and tensor nets scaled and learned but tied geometry to discrete, jumpy inputs (bond-type one-hots, grid bins) that made the energy discontinuous and forbade force training; the gradient-domain kernel was energy-conserving but didn't scale. The discontinuity all traced to filters being lookup tables over a discrete index, so I replaced the filter tensor with a filter-generating network of the continuous displacement — a continuous-filter convolution — and made it feature-wise to stay cheap, leaving cross-channel mixing to atom-wise dense layers. Feeding the filter the invariant distance (not the raw vector) gave invariance, but a near-linear filter net at init made all channels identical and stalled training, so I expanded the distance in a Gaussian RBF bank to decorrelate the channels. Twice-differentiability forced a C^∞ activation, so softplus shifted to pass through zero. Residual interaction blocks with unshared weights let a deep stack build many-body representations out of strictly radial pairwise filters, and a cosine cutoff folded into the filter kept the energy smooth as neighbors cross the cutoff boundary. Forces as the analytic energy gradient, trained with an energy-plus-force loss weighted by ρ, discharged all three physics constraints at once. And the identical geometric core — distance, Gaussian expansion, learned cosine-cut feature-wise filter, summed messages, residual blocks, pooled readout — transfers directly to encoding protein structure from alpha-carbon coordinates, needing only a lazy linear embedding over node features and a supplied spatial neighbor graph, producing rotation-invariant node and graph embeddings.

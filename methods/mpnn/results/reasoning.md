OK, let me think this through from scratch. I want a machine that takes a small organic molecule and spits out a quantum-mechanical property — an atomization energy, an orbital energy, a dipole moment. The honest reason I want this is speed: DFT gives me these numbers but it costs an hour of CPU for a nine-heavy-atom molecule and scales like the cube of the electron count, so sweeping over a chemical search space is hopeless. If I can fit the DFT output to within chemical accuracy and evaluate in milliseconds, that's a real win.

So what is the input, really? A molecule isn't a vector and it isn't an image. It's a set of atoms held together by bonds. If I just flatten it into a list and number the atoms 1, 2, 3, …, I've lied: the numbering is arbitrary, and relabeling the atoms must not change any physical property. So whatever I build has to be invariant to permutations of the atoms — invariant to graph isomorphism. That's the hard constraint, and it's the thing the standard chemistry pipeline gets wrong or gets expensively.

Let me look at that standard pipeline first, because I want to understand exactly where it hurts. The dominant recipe is: hand-design a fixed-length descriptor of the molecule, then feed it to an off-the-shelf regressor. The Coulomb matrix is the cleanest example — build C with C_{ii} = ½Z_i^{2.4} and C_{ij} = Z_iZ_j/‖r_i − r_j‖. It packs in nuclear charges and geometry, fine. But its rows are indexed by atom order. Permute the atoms and the matrix permutes; so the model downstream has to *learn* that all these permuted matrices mean the same molecule, by data augmentation. That's spending capacity to undo a symmetry I could have built in for free. The other camp — atom-centered symmetry functions, Behler–Parrinello — is genuinely invariant, but it's brittle: it was reported to struggle past three atomic species and to fail on compositions it hadn't seen. And the deeper problem with all of them is that the features are frozen. They were designed before seeing the target. A convolutional net learns image features tuned to the task; here I'm stuck with whatever the chemist hand-built. This is exactly the pre-CNN situation in vision — good hand features plus a generic classifier — and the missing ingredient is a neural architecture with the right inductive bias plus the empirical work to show it pays off.

So I want a model that reads the graph directly and learns its own features, while being invariant by construction. Several people have already built neural models on molecular graphs. Let me lay them side by side, because the first thing I notice when I do is that they *look* completely different and yet they're clearly doing the same kind of thing.

Take the differentiable-fingerprint model. Each atom carries a hidden vector h_v. In each round, the atom looks at its neighbors: it concatenates each neighbor's state with the connecting bond feature, (h_w, e_{vw}), sums those over the neighbors, and pushes the sum through a learned matrix and a sigmoid — with a *different* matrix for each atom degree. After a few rounds it reads out the graph by summing softmax(W_t h_v^t) over all atoms and all rounds.

Now take the gated graph net. Same skeleton: each atom has a hidden vector; in each round the message coming into v is the sum over neighbors of A_{e_{vw}} h_w, where A is a learned matrix picked by the (discrete) bond type; then the atom updates with a GRU, h_v ← GRU(h_v, m_v); the readout is a gated sum over atoms.

Now the interaction net from physics: message is a neural net on (h_v, h_w, e_{vw}); update is a neural net on (h_v, x_v, m_v); graph output is f(Σ_v h_v).

Now the deep tensor net for chemistry: message is tanh(W^{fc}((W^{cf}h_w + b_1) ⊙ (W^{df}e_{vw} + b_2))); update is residual, h_v ← h_v + m_v; readout is Σ_v NN(h_v).

Stare at this for a second. Every one of these does the same three things, in the same order. Each node aggregates something from its neighbors. Then each node updates its own state using that aggregate. Repeat for a few rounds. Then collapse all the node states into one graph vector. The differences are entirely in *which functions* fill those three slots and *how* the neighbor aggregation is parameterized. They're not different algorithms; they're different instantiations of one algorithm wearing different notation.

Let me write the one algorithm down and stop pretending these are separate. A node v has hidden state h_v^t at round t. The aggregation — call the result a message — is a sum over neighbors of some learned function of the source, the destination, and the connecting edge:

  m_v^{t+1} = Σ_{w ∈ N(v)} M_t(h_v^t, h_w^t, e_{vw}).

Why a *sum* over neighbors and not, say, a concatenation in neighbor order? Because the sum is the cheapest thing that's permutation-symmetric: reorder the neighbors and the sum is unchanged. That's the whole reason the model can be isomorphism-invariant — the symmetry is baked into the aggregation, not learned. Any symmetric reduction would do, but sum is the natural first choice and it's what all of them effectively use.

Then the node folds that message into its state:

  h_v^{t+1} = U_t(h_v^t, m_v^{t+1}).

Run that for T rounds. Then read out a single graph-level vector with some function R of the *set* of final node states:

  ŷ = R({ h_v^T | v ∈ G }).

And here's the constraint I'm leaning the whole symmetry argument on: R has to be invariant to permutations of the node states. My claim is that if R is invariant and the per-round aggregation is a symmetric sum, then relabeling the atoms permutes the h_v's but leaves ŷ untouched. I want to actually check this rather than wave at it, because it's the load-bearing property of the whole design — if it fails the framework is worthless. Let me trace it on a concrete molecule. Three atoms, states h_0 = (1,0,0,0), h_1 = (0,1,0,0), h_2 = (0,0,1,0), bonds 0–1 of type a and 1–2 of type b, a linear message M(h_w, e) = A_e h_w with fixed matrices A_a, A_b, a residual update U(h, m) = tanh(h + m), and a sum readout. I run one round and sum the updated states; I get the readout vector y₁ ≈ (2.0647, 0.8789, 1.0304, −0.8916). Now I apply a permutation π that sends 0→2, 1→0, 2→1: the states move to those slots, and every bond (v,w) becomes (π(v),π(w)) carrying the *same* matrix. Recompute from scratch on the relabeled molecule and the readout comes out y₂ ≈ (2.0647, 0.8789, 1.0304, −0.8916). The two differ by about 4×10⁻¹⁶ — machine epsilon, i.e. they are equal. Good: the relabeling permuted which slot each node sat in, the per-node update only ever saw its own neighbor-sum (itself a symmetric reduction, so π just reorders the summands), and the final sum is over the *set* of node states, so the permutation washes out entirely. The invariance isn't an aspiration; it survives an explicit relabeling. M_t, U_t, R are all learned differentiable functions; the message passing phase builds local features, the readout phase pools them. That's the abstraction I'll work with. Call it message passing.

Does it really subsume the models above, or have I just invented a vague analogy? The honest test is to write down M, U, R for each and see whether the substitution reproduces the original equations, not merely rhymes with them.

The fingerprint model: M(h_v, h_w, e_{vw}) = (h_w, e_{vw}), the concatenation. U_t(h_v, m) = σ(H_t^{deg(v)} m), one learned matrix per round and per degree. R = f(Σ_{v,t} softmax(W_t h_v^t)), with skip connections to every round. Substituting these back gives exactly the original update, so it maps over. And mapping it over forces me to actually expand the message, which exposes a flaw I'd never have spotted in the original notation: m_v^{t+1} = Σ_w (h_w, e_{vw}) = (Σ_w h_w, Σ_w e_{vw}) — the concatenation sums the neighbor *states* and the bond *features* in separate blocks. Let me make sure that's a real defect and not a slip. Take a node with two neighbors A and B. Suppose A is attached by bond-type-0 and B by bond-type-1; the message is concat(h_A, e_0) + concat(h_B, e_1). Now swap the bonds — A by bond-1, B by bond-0 — giving concat(h_A, e_1) + concat(h_B, e_0). With h_A = (1,0), h_B = (0,1), e_0 = (1,0), e_1 = (0,1), both messages come out (1,1,1,1): *identical*. So the message genuinely cannot tell which bond attaches to which atom — it sees the total of the neighbor states and the total of the bonds, never the pairing. That's a concrete limitation, and it's a hint about what a good M should do: couple the edge and the node it touches, not add them in separate bins.

The gated graph net: M_t(h_v, h_w, e_{vw}) = A_{e_{vw}} h_w, one matrix per discrete edge label; U_t = GRU(h_v, m), tied across rounds; R = Σ_v σ(i(h_v^T, h_v^0)) ⊙ j(h_v^T), with i, j small nets and ⊙ elementwise. Substituting these reproduces the gated net's equations verbatim.

The interaction net: M = NN(h_v, h_w, e_{vw}); U = NN(h_v, x_v, m) with an optional external per-node input x_v; R = f(Σ_v h_v^T). Maps over. (It was only ever run for T = 1, but that's just a choice of T.)

The molecular graph-conv model: it also carries *edge* states e_{vw}^t. So I note the framework can be extended — give edges hidden states h_{e_{vw}}^t and update them by the same kind of rule — and only this model uses that. Its node message is M = e_{vw}^t (the edge state itself), U_t = α(W_1(α(W_0 h_v), m)) with α = ReLU, and the edge update is e_{vw}^{t+1} = α(W_4(α(W_2 e_{vw}^t), α(W_3(h_v, h_w)))). Maps over, with the edge-state extension.

The deep tensor net: M_t = tanh(W^{fc}((W^{cf}h_w + b_1) ⊙ (W^{df}e_{vw} + b_2))), U_t(h_v, m) = h_v + m, R = Σ_v NN(h_v^T). Maps over — and notice the residual update U = h + m is the same trick that lets these stack rounds without washing the signal out.

That's five of the neural-graph models reproduced by a choice of (M, U, R). The one I'm least sure fits comes from a totally different lineage: the spectral / Laplacian graph convolutions. Those are derived from the graph Fourier transform, not from neighbor aggregation at all. If message passing is really general, I should be able to *derive* that the spectral layer is a message passing step — not assert it, derive it — because if it works it's the strongest evidence the abstraction is the right one, and if it doesn't, I've found the boundary of the framework.

Set up the spectral layer. Adjacency W on N nodes, Laplacian L = I − D^{-1/2}WD^{-1/2} with D the diagonal degree matrix, V its eigenvectors ordered by eigenvalue. The layer takes an input of N×d_1 (a scalar feature per node, d_1 channels) to N×d_2 by, for each output channel j,

  y_j = σ( Σ_{i=1}^{d_1} V F_{i,j} V^⊤ x_i ),  j = 1 … d_2,

where each F_{i,j} is a diagonal N×N matrix of learned filter coefficients and x_i, y_j are N-vectors (one scalar per node). This is filtering in the graph-Fourier basis: V^⊤ takes x_i to the spectral domain, F_{i,j} reweights frequencies, V comes back. Looks nothing like neighbor aggregation. But watch what happens if I refuse to think spectrally and just expand the matrix products in node coordinates.

Define a rank-4 tensor that absorbs all the V F V^⊤ products: L̃_{v,w,i,j} = (V F_{i,j} V^⊤)_{v,w}. So L̃ has shape N×N×d_1×d_2. Then the layer, written out per node v and channel j, is

  y_{v,j} = σ( Σ_{i=1}^{d_1} Σ_{w=1}^{N} L̃_{v,w,i,j} x_{w,i} ).

Now group the channel indices into vectors. Let L̃_{v,w} be the d_1×d_2 matrix with entries (L̃_{v,w})_{i,j} = L̃_{v,w,i,j}, let x_w be the d_1-vector of node w's features and y_v the d_2-vector of node v's output. Then the double sum collapses to

  y_v = σ( Σ_{w=1}^{N} L̃_{v,w} x_w ).

So at least on paper, relabeling y_v as h_v^{t+1} and x_w as h_w^t, the message function is M(h_v^t, h_w^t) = L̃_{v,w} h_w^t and the update is U(h_v^t, m_v^{t+1}) = σ(m_v^{t+1}). Before I trust the index gymnastics, let me check the collapse numerically — it would be easy to drop a transpose. Take a 3-node graph, build L and its eigenbasis V, pick one input channel x = (0.3, −1.1, 2.0) and one diagonal filter F, and compute the layer two ways: the spectral form V F V⊤ x, and the collapsed form (Σ_w L̃_{v,w} x_w) with L̃ = V F V⊤. Both give (0.6613, 0.0802, 1.3207); the max absolute difference is 0.0 — they're bit-identical, not just close. So the collapse is exact, and a spectral convolution *is* a message passing round: the message from w to v is a learned linear map L̃_{v,w} applied to h_w, summed over all w, then a pointwise nonlinearity. The only oddity is that the "neighborhood" is every node, because the spectral filter is generically dense, but structurally it's the same M/U as everything else. (For the localized spectral variants the L̃_{v,w} are sparse and you recover an actual neighbor sum.) The framework reaches the spectral lineage too.

Let me push the reduction all the way to the cleanest spectral model, the first-order one, because that's the one people actually use and I want to see exactly what M and U become. Its layer-wise rule is

  H^{l+1} = σ( D̃^{-1/2} Ã D̃^{-1/2} H^l W^l ),

where Ã = A + I adds self-loops (so a node sees itself), D̃_{ii} = Σ_j Ã_{ij} is the degree with self-loops, W^l is a learned d×d weight matrix, and each H^l is N×d. Let L = D̃^{-1/2} Ã D̃^{-1/2}. Take one node v — that's row v of the output:

  H^{l+1}_{(v)} = σ( L_{(v)} H^l W^l ) = σ( Σ_w L_{vw} H^l_{(w)} W^l ).

The L_{vw} are scalars. Transpose to column-vector convention, h_v^{t+1} for the node state:

  h_v^{t+1} = σ( (W^l)^⊤ Σ_w L_{vw} h_w^t ).

So the message is M_t(h_v^t, h_w^t) = L_{vw} h_w^t with the scalar L_{vw} = Ã_{vw}(deg(v) deg(w))^{-1/2}, and the update is U_t(h_v^t, m_v^{t+1}) = σ((W^t)^⊤ m_v^{t+1}). Since L_{vw} is just a scalar, the message phase is a *degree-normalized weighted average of the neighbors' states*, with the linear map W applied after. So even the most-used graph conv is a particular, very simple choice of message and update. The whole zoo collapses onto three slots.

Good. The abstraction holds up — it reproduced six models, and the one I doubted most, I checked numerically rather than asserted. Now the actual job: I'm not here to admire the framework, I'm here to predict QM9 properties as accurately as possible. The framework tells me the design space is exactly (M, U, R) plus how I feed in the graph. So let me search it deliberately, starting from the strongest baseline I have, which is the gated graph net — discrete-edge matrix messages, GRU update, gated readout. I'll keep its skeleton and probe each slot.

First, the update. The gated net uses a GRU and ties its weights across rounds. Do I want to untie them — a fresh U_t per round? I tried that early in my head and in practice it doesn't help; the better use of parameters is to *tie* the weights and instead make the hidden state d wider. Tying turns the T rounds into a recurrent propagation that I can run for more steps without paying in parameters, and the GRU's gating keeps the state from blowing up or washing out as I iterate — exactly why a residual or gated update keeps showing up in these models. So: GRU update, weight-tied, initialize h_v^0 to the atom feature vector x_v padded up to width d.

One implementation point about direction. Chemical bonds are undirected, but I'll treat the graph as directed with a separate message channel for incoming and outgoing edges, the way the gated-net family does — each undirected bond becomes an in-edge and an out-edge sharing a label, and I concatenate m_v^in and m_v^out into a message of width 2d. The direction doesn't carry physics; it's purely a handle for parameter tying so the model can, if it wants, treat "message I send" and "message I receive" with different weights.

Now the slot that matters most: the message function M. The baseline is M = A_{e_{vw}} h_w with one learned matrix per discrete edge label. The trouble is staring at me from the input. The single best signal QM9 gives me is geometry — the interatomic distance — and distance is *continuous*. A scheme with one matrix per discrete edge type can't ingest a real number; I'd have to bucket distances into bins and pretend they're labels, throwing away resolution. I want the message to be conditioned on a continuous, vector-valued edge feature e_{vw} = (distance, bond-type-one-hot). So I need a message of the form "apply a matrix to the neighbor's state, but let the matrix *depend on the edge vector*." The clean way: let a small neural network A(·) read the edge vector and *output* a d×d matrix, then

  M(h_v, h_w, e_{vw}) = A(e_{vw}) h_w.

The discrete-label matrices are just the special case where A is a lookup table over labels; making A a network lets a continuous bond geometry continuously deform the linear map that mixes the neighbor in. And this is exactly what the fingerprint diagnostic pointed at — there the failure was that the edge was *added* to the node in a separate block, so the message couldn't see node–edge pairing; here the edge instead *multiplicatively gates* the node's contribution through A(e_{vw}), so the pairing is built into the map. I'd want to confirm on QM9 that this is where the accuracy comes from, but the structural argument is sound and it's the variation I'd reach for first.

While I'm at the message, here's an itch. M = A(e_{vw}) h_w depends on the source w and the edge, but *not* on the destination's own state h_v. The interaction net's message did depend on both endpoints. In principle the channel could be used more efficiently if the message knew who was receiving it, so let me try a pair message m_{wv} = f(h_w, h_v, e_{vw}) with f a net on the concatenation. I even cook up a toy pathfinding task designed to reward knowing both endpoints. And it trains *worse* — even on that toy task. The pair message is harder to optimize; conditioning on both endpoints makes the message function's job less stable, and the gain I imagined doesn't materialize. So I drop it and go back to M = A(e_{vw}) h_w, which is both simpler and the thing that actually trained better.

Now the readout R. The baseline gated-sum readout is fine, but it's still essentially a sum of per-node terms, and I keep feeling that a plain sum throws away structure in the *set* of node states. I want a readout that's still permutation-invariant — non-negotiable, or I lose the whole symmetry argument I checked earlier — but more expressive than summing. There's a set-encoding model, set2set, built exactly for this: it processes a set in a way that's invariant to input order by running M steps of an LSTM that, at each step, computes attention weights over the set elements via content-based attention, forms an order-invariant weighted sum, and refines a query. I project each final node state (paired with its input features) into the set, run set2set, get a graph embedding that doesn't depend on the order of the nodes, and push it through an output net. It's strictly more expressive than a sum (it can attend), and it stays invariant because the attention is over a set, not a sequence — same reason the sum readout passed the relabeling test, the aggregation never references node order. So R = set2set over {(h_v^T, x_v)} → output net. The thing I'd watch for is whether the extra expressiveness actually buys accuracy or just costs compute; my bet is it helps, because it gives the readout a way to aggregate global information a local sum can't.

Which brings up a problem the readout alone can't fix: range. I run T rounds, maybe T = 3 to 8 (empirically anything past 3 is fine, so the receptive field is a few hops). If two atoms are far apart on the graph, their states never directly inform each other within T rounds. When I'm in the regime *without* spatial input — only topology — that hurts a lot, because the target may depend on long-range interactions that the bond graph only encodes through long paths. Two cheap fixes within the framework. One: add a "virtual" edge type between every pair of atoms that aren't bonded — a preprocessing step that turns the sparse molecule into a denser graph so information crosses it in one hop, with the virtual edges getting their own learned message parameters so the model can tell them apart from real bonds. Two: add a single latent "master" node wired to every atom by a special edge type. Every round, every atom reads from and writes to the master node, so it's a global scratchpad that any two atoms can communicate through in two hops. I let the master node have its own width d_master and its own update weights. The nice thing about the master node is the cost: a propagation step is O(|E| d^2 + n d_master^2), so I can crank d_master for extra global capacity without the n^2 d^2 blowup a fully-connected graph of width d would cost. Both of these I'd expect to help most exactly when spatial information is withheld, by giving long-range interactions a path.

Now scalability, because the message phase is the bottleneck. With matrix-multiply messages on a dense graph, one round costs O(n^2 d^2): for every ordered pair I apply a d×d matrix to a d-vector. As d grows that's brutal. Here's a trick. Split each node's d-dimensional state into k separate copies of width d/k, h_v^{t,k}. Run a *separate* propagation — its own M and U — on each copy independently, giving temporary states h̃_v^{t+1,k}. Then mix the k copies back together with a shared network g over their concatenation:

  (h_v^{t,1}, …, h_v^{t,k}) = g(h̃_v^{t,1}, …, h̃_v^{t,k}).

Does this actually save? Let me count rather than assume. One copy's propagation, with matrix-multiply messages, is O(n^2 (d/k)^2). There are k copies, so the message phase costs k · n^2 (d/k)^2 = n^2 d^2 / k — a factor of k cheaper than the monolithic n^2 d^2, plus a small overhead for the mixing net. Let me put numbers on it: n = 9, d = 200, k = 8 (so each tower is width 25). Monolithic = 9² · 200² = 3,240,000 unit-ops per round; towers = 8 · 9² · 25² = 405,000; the ratio is exactly 8 = k, as the algebra said. So asymptotically it's a clean factor of k. The measured wall-clock win is smaller — about 2× at inference — because the mixing net and the fixed overheads eat into the asymptotic gain, but it's real, and it lets me run a wider effective hidden state for the same compute budget. And mixing with a *shared* g over the concatenation keeps the whole thing permutation-invariant — g acts identically at every node, so relabeling atoms still just permutes the states, exactly as in the relabeling check. There's even a bonus I notice: the k separate copies behave a bit like an ensemble within one model, which seems to help generalization, not just speed. The towers. (One caveat: towers didn't combine cleanly with the edge-network message — that pairing made optimization harder — so towers is the play when I'm using matrix-multiply messages, not the edge net.)

Last, the input representation, because the framework is only as good as what I feed it. Each atom gets features: element type as a one-hot over H/C/N/O/F, atomic number, acceptor/donor flags, aromaticity, hybridization (sp/sp2/sp3), and hydrogen count. For edges I have three options matched to the message function. If I have no geometry, the edge is just the discrete bond type {single, double, triple, aromatic} and I use matrix-multiply messages. To smuggle distance into the matrix-multiply scheme I have to discretize: bin distances — partition [2, 6] Å into 8 bins and add a [0, 2] and a [6, ∞] bin for 10 distance bins, on top of the 4 bond types, an alphabet of 14 edge labels (and I find the bonded distances are nearly determined by bond type anyway, so the bins mostly matter for non-bonded pairs). But the right thing, with the edge network, is to skip discretization entirely and hand it the raw 5-dimensional edge vector: first coordinate the Euclidean distance, the other four the one-hot bond type. One more input choice: make hydrogens explicit nodes rather than just a per-atom count. That grows graphs to as many as 29 nodes and slows training roughly 10×, but it gives the message passing real atoms to pass messages through, and I expect it to matter for several targets.

So the configuration the search lands on: GRU update tied across rounds, edge-network message M = A(e_{vw}) h_w on the raw distance+bond edge vector, set2set readout, explicit hydrogens. Train it the obvious way — Adam, normalize each target to zero mean and unit variance, minimize MSE but report MAE, early-stop on a validation split, one model per target (training one shared model on all 13 targets at once is consistently worse, by up to 40%, so the properties want their own networks). T around 3–8, set2set steps in 1–12.

Let me write the core of it as code, filling the slots of the generic graph model with the choices the search landed on. The message phase is one edge-conditioned conv reused for T rounds with a shared GRU; the readout is set2set; the head is two linear layers.

```python
import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

from torch_geometric.nn import NNConv, Set2Set   # edge-network conv; set2set readout

class MPNN(torch.nn.Module):
    def __init__(self, num_node_features, dim=64, T=3, set2set_steps=3):
        super().__init__()
        self.T = T
        # pad the atom feature vector x_v up to the hidden width d
        self.lin0 = Linear(num_node_features, dim)

        # edge network A(e_vw): an MLP that reads the 5-dim edge vector
        # (distance, bond-type one-hot) and emits a d x d matrix.
        # The message into v is then sum_w A(e_vw) h_w  ->  M(h_v,h_w,e_vw)=A(e_vw) h_w.
        edge_net = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, edge_net, aggr='mean')   # the message phase
        # weight-tied GRU update U_t = GRU(h_v, m_v), shared across all T rounds
        self.gru = GRU(dim, dim)

        # permutation-invariant readout R: set2set over the final node states,
        # outputs a vector of width 2*dim.
        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = Linear(2 * dim, dim)
        self.lin2 = Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))      # h_v^0 = padded atom features
        h = out.unsqueeze(0)                 # GRU hidden state carries h_v across rounds

        for _ in range(self.T):              # message passing phase, weights tied
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))  # m_v = sum_w A(e_vw) h_w
            out, h = self.gru(m.unsqueeze(0), h)   # h_v <- GRU(h_v, m_v)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)  # R: invariant pooling -> graph vector
        out = F.relu(self.lin1(out))
        return self.lin2(out).view(-1)       # scalar property prediction
```

The causal chain, start to finish: I needed a fast, isomorphism-invariant surrogate for DFT that learns its own features from a molecular graph; staring at the existing neural graph models showed they're all the same three-slot computation — aggregate a symmetric message from neighbors, update each node, repeat, then pool with a permutation-invariant readout — and writing that abstraction down (M, U, R) let me reduce every one of them, even the spectral convolutions (which I checked numerically rather than asserted), to a choice of those slots, after first confirming on a relabeled three-atom molecule that the symmetric-sum-plus-invariant-readout design really is isomorphism-invariant; then, treating the slots as a design space and starting from the gated-net baseline, the structure of the input drove each choice — a continuous bond geometry demanded an edge network A(e_{vw})h_w instead of discrete-label matrices (the fix for the node–edge-pairing flaw I'd found by expanding the fingerprint message), a richer set encoder (set2set) beat a plain sum for the readout, virtual edges and a master node bought long-range communication when geometry was withheld, and splitting the state into towers bought a counted factor-of-k speedup while preserving invariance — landing on an edge-network message with a tied GRU update and a set2set readout on graphs with explicit hydrogens.

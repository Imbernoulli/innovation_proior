I'm handed a backbone — for each of `L` residues the coordinates of `N, CA, C, O` — and I have to produce a sequence that would fold into it. Folding goes sequence → structure; I'm doing the inverse, structure → sequence. Let me be honest about what makes this hard before I reach for any machinery. The map is degenerate: lots of sequences fold to nearly the same shape, so there isn't one right answer, and pretending there is one and regressing to it would be wrong in principle. What I actually want is a distribution over compatible sequences. The couplings are awkward too — residues that are far apart along the chain can be touching in space and have to be chosen together, while sequence-adjacent residues might point into completely different environments. And the only thing that can legitimately determine the sequence is the *geometry*: if I rotate or translate the whole molecule the answer can't change, so whatever I compute has to be invariant to rigid-body motion. So the target is `p(s | x)`, invariant, modeling the joint over all `L` positions, on proteins of hundreds of residues, on one GPU.

The physics way to do this is Rosetta: write down an energy function — van der Waals, solvation, hydrogen bonds, electrostatics, rotamer statistics — and Monte-Carlo over amino-acid identities and side-chain rotamers until the energy is low. It bakes in real structural-biology knowledge, but the energy is hand-tuned and approximate, the side-chain packing search is slow, and it recovers the native amino acid only about a third of the time. I'd rather *learn* the energy. There's a whole Protein Data Bank of solved structures; let me learn `p(s | x)` directly and let the network discover its own features and its own effective scoring instead of me guessing the weights.

How do I feed a 3D structure to a network in an invariant way? The natural object is a graph. Pairwise distances between atoms are invariant to rotation and translation by construction — that's the cleanest source of invariance I have — so a graph whose edges carry distance-derived features is automatically invariant without me having to build equivariant machinery. And I shouldn't make it a complete graph: dense attention or dense message passing over all pairs is `O(L^2)` memory and compute, which won't fit for a long protein. But I know from coevolution work that the contacts that carry the energetics are *spatially local* — a residue's relevant partners are the handful nearest it in space, and that count doesn't grow with chain length. So I'll restrict to a `k`-nearest-neighbor graph over the `Cα` atoms, `k` around 30. That makes everything `O(L·k)` — linear in length — and throws away almost nothing that matters.

For the computation on the graph, the message-passing framework is exactly right. Each round, form a message into every node by summing a learned function over its neighbors, then update the node:

```
m_i = sum_{j in N(i)} M(h_i, h_j, e_ij),
h_i' = U(h_i, m_i).
```

The neighbor sum is symmetric, so the whole thing is invariant to how I happen to index the graph — the permutation invariance I want falls out for free. Now I need to actually decide the three things this framework leaves open: what the features `e_ij` and `h_i` are, what the per-round update is, and how the output is produced. That's the whole design.

Start with the cleanest version that already exists and find where it hurts, because that tells me what to fix. The strong prior here is a Structured Transformer for this task: cast design as autoregressive language modeling conditioned on the structure graph,

```
p(s | x) = prod_i p(s_i | x, s_{<i}),
```

with a graph-restricted encoder–decoder. The encoder is sparse self-attention over the `k`-NN neighbors building a sequence-independent representation of structure; the decoder predicts each amino acid left-to-right, `N → C`, masking so position `i` sees the whole structure but only the already-emitted amino acids `s_{<i}`. Its node features are the backbone dihedrals `(sin, cos)(φ, ψ, ω)`; its edge features concatenate a distance RBF, a relative orientation encoded as the quaternion of the rotation between local backbone frames, and a sinusoidal encoding of the offset `i − j`. Three encoder and three decoder layers, hidden 128, the Transformer warmup schedule, dropout, label smoothing. It beats the profile baselines and beats Rosetta on both recovery and speed. Good. So what's left on the table?

The first thing that bothers me when I trace the encoder is that it refines only the *node* embeddings. Look at where the information lives: the structural signal — which residues are close, in what relative pose — is in the *edges*. The nodes start as a thin per-residue summary (just the dihedrals). The encoder then does several rounds of attention that update node states, but the edge features were computed once at featurization and are frozen forever after. That's a waste. The relational features are the most informative part of the input, and I'm denying them any chance to be refined by context. A contact between two residues *means* different things depending on the rest of the local environment, and a static edge feature can't express that. So one move is already clear in outline: let the edges update too, the way the molecular-graph-convolution line did — maintain an edge state and refresh it each round using the freshly updated node states. I'll come back and make that concrete.

The second thing is the decoding order. Strictly left-to-right means at inference a position can only ever condition on lower-index residues. But think about what I actually want to *do* with this model in a design loop. Often I have some positions fixed — an active site, an interface, a motif I'm keeping — and I want to design the rest *conditioned on* those fixed ones, regardless of where they sit in the sequence. With an `N → C` factorization I can't use a high-index fixed residue as context for a low-index designed one; the causal mask forbids it. The chain direction isn't sacred here — any ordering of the positions is an equally valid factorization of the same joint distribution, since `p(s | x) = prod_t p(s_{o_t} | x, s_{o_{<t}})` holds for *any* permutation `o`. So locking in `N → C` is an arbitrary, restrictive choice. I'll want order-agnostic decoding. Hold that too.

Third, a quieter observation: the Structured Transformer carries multi-head attention and explicit orientation/quaternion edge features, but the diagnostic ablation on this very data showed a plain MLP message aggregator `Δh_i = sum_j MLP(h_i, h_j, e_ij)` reaching *better* held-out perplexity than attention, and showed distance-only edges already beating the profile baselines with orientation adding only a little. Attention seems to be overfitting at this data scale — the extra capacity to weight neighbors learnably isn't paying for itself, the simpler symmetric sum generalizes better. That's telling me to drop attention in favor of a straight MLP-over-neighbors message, and to lean on *distances* rather than hand-built orientation frames.

And fourth, the deployment reality: the model is trained on crystal-exact coordinates, but in a real design pipeline the backbone I feed it is *predicted* or *hallucinated* or low-resolution — it's wrong at the sub-Ångström level and sometimes more. A model that has only ever seen perfect coordinates will have latched onto fine geometric detail that simply won't be there at deployment. That's a train/test distribution shift waiting to bite. Park that; it'll want a regularizer.

Let me build the featurization first, because the "drop orientation features, lean on distances" decision needs to be made honest. If I'm going to throw away the explicit quaternion orientation encoding, I have to be sure I'm not throwing away the orientation *information*. Here's the thing: orientation is recoverable from distances if I use *enough* of them. A single `Cα–Cα` distance per edge is not locally informative — two neighbors at the same distance could be on the same side or opposite sides of a residue, and the scalar can't tell them apart. But if for each edge I take the pairwise distances among a small set of backbone atoms on both residues, the relative rigid-body pose of the two residues is essentially pinned down — it's a little distance-geometry problem, and the cross-distances between the two atom triads encode the relative rotation implicitly. So instead of building local frames and computing quaternions, I'll just compute *all* the inter-atomic distances between the atoms of residue `i` and the atoms of residue `j`. Distances are invariant by construction, so this stays fully invariant with zero frame bookkeeping, and it captures orientation through the *pattern* of which atoms are near which.

Which atoms? I have `N, CA, C, O`. Side-chain identity is what I'm predicting, so I don't have side-chain atoms — but the *direction* a side chain would point is a strong cue about a residue's environment (buried vs exposed, pointing at a neighbor or away). I can get a proxy for free: place a virtual `Cβ` at the ideal tetrahedral position from the backbone. With `b = CA − N`, `c = C − CA`, `a = b × c`,

```
Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA.
```

Those constants are just the tetrahedral geometry that puts `Cβ` where it sits in a real residue. Now I have five atoms per residue: `N, CA, C, O, Cb`. For an edge `(i, j)` I take the distances between each atom of `i` and each atom of `j` — but I should be deliberate about which ordered pairs to include so the count is fixed and the set is rich. Twenty-five ordered atom pairs covers the full cross-product richly: `Ca-Ca`, the four same-atom pairs `N-N, C-C, O-O, Cb-Cb`, and the cross pairs in both orientations (`Ca-N` and `N-Ca` are different edges of the local frame pattern, so keeping both directions helps the network read the relative pose). Twenty-five distances per edge, then.

A raw distance is a bad thing to hand a linear layer — it would have to learn a bump function of a scalar to react to "this is a contact at ~5 Å." So expand each distance into a bank of Gaussians at fixed centers,

```
RBF(D)_n = exp( -((D - mu_n)/sigma)^2 ),   mu_n = linspace(2, 22, 16),  sigma = (22 - 2)/16,
```

sixteen centers spread from 2 to 22 Å — contact range out to "distant but still in the neighborhood." That turns "distance ≈ `mu_n`" into a near-one-hot soft code, so the first linear layer can act like a lookup over distance bins. Sixteen RBFs times twenty-five atom pairs is 400 edge features from distances.

I also want sequence position in the edge, because two spatial neighbors that are also sequence-adjacent are a different situation from two that are 200 residues apart. The chain has a direction (`N → C`) but no canonical origin, so the right thing is a *relative* encoding of the offset `i − j`, keeping its sign. Multi-chain design adds one more wrinkle: an offset across chains is not the same kind of object as an offset within one chain. So I take the residue-index difference, clip it to a fixed window `[-32, 32]`, reserve one extra bucket for cross-chain neighbors, one-hot that bucket, and learn a 16-dimensional positional embedding from it. Concatenate those 16 positional features onto the 400 distance features, then embed the `416`-dim edge feature to the hidden width with a linear map and a LayerNorm.

Now the nodes. The Structured Transformer put dihedrals into the nodes. But I've just decided the edges carry the rich, invariant geometry; the dihedrals are a weaker per-residue summary of the same backbone, and they'd be a second, redundant channel. Let me push the principle to its limit: put *no* geometry directly in the nodes — initialize the node states to **zero** — and force all structural information to enter through the edges and propagate into the nodes via message passing. This makes the model maximally reliant on the relational, invariant signal, which is exactly the signal that transfers to unseen folds. It feels aggressive, but it's consistent: if the edges really are a complete local description, the encoder can build all the node state it needs from messages, and I avoid handing the model a per-residue feature that might encode fold-specific quirks. Zero node init, geometry only in edges.

Time to make the per-layer update concrete, and this is where the "edges should update too" decision becomes real. For a node message I gather, for each edge `(i, j)`, the triple `[h_i, h_j, e_ij]` — the current center-node state, the neighbor-node state, and the edge state — and run it through a small MLP. I'll use three linear layers with a nonlinearity between them, `W3(g(W2(g(W1(·)))))`, with `g` a smooth nonlinearity (GELU — smoother than ReLU, the standard transformer-era choice and it costs nothing). Sum the per-neighbor messages, residual-add into the node, LayerNorm, then a position-wise feed-forward (a wider 4× MLP, GELU) with its own residual and LayerNorm — the Transformer block shape, but with the attention swapped for the symmetric MLP-over-neighbors that the ablation said generalizes better.

There's a normalization subtlety in the neighbor sum I have to get right or LayerNorm will fight me. If I literally *sum* over neighbors, the magnitude of `dh_i` grows with the number of neighbors, which varies (padded positions, short proteins, edges of the graph), so the activation scale into the residual/LayerNorm would be inconsistent. I want an approximate *mean*, not a sum. The cheap, fixed fix is to divide the summed message by a constant `scale` of the order of the neighborhood size — `scale = 30`. That keeps `dh_i` at the magnitude of a single message across the usual sparse-neighborhood regime, so the residual and LayerNorm see a stable scale. (A true per-node mean would divide by the actual unmasked degree; the fixed constant is simpler and, with masking zeroing out padded neighbors before the sum, close enough.) So:

```
dh_i = (1/scale) * sum_{j in N(i)} m_ij,   then  h_i <- LayerNorm(h_i + dropout(dh_i)),
h_i <- LayerNorm(h_i + dropout(FFN(h_i))).
```

And now the edge update, the piece the Structured Transformer didn't have. After the nodes have been refreshed, recompute the edge's input triple with the *new* node states, `[h_i, h_j, e_ij]`, run it through a second three-layer MLP `W13(g(W12(g(W11(·)))))`, and residual-add into the edge with a LayerNorm:

```
e_ij <- LayerNorm(e_ij + dropout( W13(g(W12(g(W11([h_i, h_j, e_ij]))))) )).
```

That's the structural improvement made concrete: each round, nodes aggregate from neighbors and edges, *then* edges are re-derived from the refreshed nodes, so "what this contact means" gets refined through depth instead of being frozen at featurization. Three such encoder layers — three rounds reach roughly three-hop spatial neighborhoods, which is enough for the local geometry that dominates, and three keeps it fast. Hidden 128, carried over.

Now the decoder, and here's where order-agnostic decoding has to be built carefully. The factorization `p(s | x) = prod_t p(s_{o_t} | x, s_{o_{<t}})` is valid for any permutation `o` of positions, so I'll *train* over random orders: each example, sample a random order, and predict each residue from the structure plus the amino acids that come *before it in that order*. At inference I can then decode in any order I like, including putting fixed positions first so they always serve as context. The reason this is worth the trouble is threefold: it lets a designer fix arbitrary positions and design the rest (impossible with a frozen `N → C` mask); it's a denser training signal, since each residue gets predicted from many different context subsets across epochs, a BERT-like masking effect living inside an autoregressive factorization; and because every order is a correct factorization of the same joint, averaging over random orders is a *consistent* objective, not a hack.

Implementing the order-agnostic causal mask is the fiddly bit; let me derive it rather than wave at it. I need, for each position `q`, a mask over the other positions `p` that is 1 exactly when `p` precedes `q` in the chosen order — i.e. `p` is already decoded when I predict `q`. First, the order. I have a per-position flag `chain_M` that is 1 for positions I'm designing and 0 for positions that are fixed/visible. I want fixed positions to be decoded "first" (always context) and designed positions in random order among themselves. Draw `randn ~ N(0,1)` per position and sort by `(chain_M + 1e-4) * |randn|`: fixed positions (flag 0) get the tiny key `1e-4*|randn|` so they sort to the front; designed positions (flag 1) get key `|randn|`, i.e. a random order. So `decoding_order = argsort((chain_M + 1e-4) * |randn|)` gives a random permutation with fixed positions front-loaded.

Now turn that order into a "is-`p`-before-`q`" matrix. Let `P` be the one-hot permutation matrix of `decoding_order`, `P[b, i, q] = 1` if the `i`-th decoded position is `q`. Take the strictly-lower-triangular all-ones matrix in *order space*, `Lhat = 1 - triu(ones(L, L))`, which has `Lhat[i, j] = 1` exactly when `j < i` — decode step `j` comes before decode step `i`. I need to carry this from order space (indexed by decode step) back to position space (indexed by `p, q`). Conjugating by `P` does exactly that:

```
order_mask_backward[b, q, p] = sum_{i, j} Lhat[i, j] * P[b, i, q] * P[b, j, p],
```

which in einsum is `einsum('ij,biq,bjp->bqp', Lhat, P, P)`. Read it off: the term is nonzero only when `q` is the `i`-th decoded position and `p` is the `j`-th, and `Lhat[i,j]=1` means `j < i`, i.e. `p` is decoded strictly before `q`. So `order_mask_backward[b, q, p] = 1` iff `p` precedes `q` in the order — exactly the "already-decoded predecessors of `q`" indicator I wanted. Finally I only care about the `k` graph neighbors, so gather this along the neighbor axis: `mask_attend = gather(order_mask_backward, dim=2, E_idx)`, giving for each `q` a 0/1 over its `k` neighbors marking which are already decoded.

With that mask, the decoder's information flow is: an already-decoded neighbor may contribute its *sequence* (its amino-acid embedding `h_S_j = W_s(s_j)`) on top of its structure; a not-yet-decoded neighbor contributes *structure only*, its sequence zeroed. Split the mask into backward (decoded) and forward (undecoded), `mask_bw = mask * mask_attend`, `mask_fw = mask * (1 - mask_attend)`, and build the decoder's neighbor features as

```
h_ESV = mask_bw * cat(h_V, h_E + h_S_neighbors) + mask_fw * cat(h_V, h_E, zeros),
```

so sequence flows in only from predecessors while structure flows from everyone. The decoder layers are the same message+feed-forward block as the encoder but *without* the edge update (the edges are done being refined; the decoder is consuming them), three of them. A final linear maps each node to the 21 amino-acid logits, and a log-softmax gives per-residue log-probabilities. Train with masked cross-entropy; for the smoothed training objective the implementation uses a fixed `2000.0` denominator, while the unsmoothed NLL can still be averaged over valid residues for reporting. Add label smoothing of 0.1 so the model isn't pushed to overconfidence on what is, after all, a degenerate target with many acceptable answers.

Now the parked deployment problem: train on crystal-exact coordinates, deploy on imperfect backbones. The clean fix is data augmentation in the input space — perturb the coordinates with Gaussian noise during training, `X <- X + augment_eps * N(0,1)`, applied only in training mode and only inside featurization so the *labels* are untouched (I still ask it to recover the native sequence from a slightly jiggled backbone). This deliberately destroys sub-Ångström detail that the model would otherwise overfit to, forcing it to rely on the coarser geometry that *will* be present when the backbone is predicted or hallucinated. How much noise? It's a robustness-vs-fidelity knob, so I'll train a small family at `augment_eps in {0.02, 0.10, 0.20, 0.30}` Å and choose the member according to the backbone source — smaller noise when the input backbones are near-crystal, larger noise when they're rough designed backbones. This is exactly the regularizer the distribution shift demanded.

A couple of training details that the transformer-shaped architecture pulls along. The optimizer is Adam with the Noam warmup schedule, `lr = factor * d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})`, `factor = 2`, `warmup = 4000`, `Adam(betas=(0.9, 0.98), eps=1e-9)` — warmup is what keeps the early steps of a LayerNorm/residual net from blowing up before the statistics settle. Dropout 0.1 throughout. And at sampling time I draw from a temperature-adjusted softmax; low temperature (≈ 0.1) concentrates on the high-probability amino acid for maximum recovery, higher temperature trades recovery for diversity when I want varied designs.

Let me sanity-check the whole pipeline against the invariance requirement one more time, because everything hinges on it. The node states start at zero (invariant). The edges are functions of inter-atomic *distances* (invariant) plus relative sequence offset (frame-independent). Every layer's message is an MLP of node and edge states summed symmetrically over neighbors (permutation- and frame-invariant). The decoder adds sequence embeddings, which are frame-independent. So the predicted distribution is invariant to rigid-body motion by construction, with no equivariant vector bookkeeping anywhere — I got the invariance purely from choosing distances as the geometric primitive. Good. And the cost is `O(L·k)` per layer, linear in length, fits on a GPU.

So the causal chain, end to end. I started from a degenerate, invariance-constrained `p(s | x)` over a sparse spatial graph, and the message-passing-on-a-`k`-NN-graph skeleton was forced by invariance plus the locality of contacts plus the GPU budget. The strongest prior, a Structured Transformer, left four things open: it froze the edge features, it locked decoding to `N → C`, it used attention where a plain MLP generalized better, and it trained only on perfect coordinates. Each gap drove a concrete piece. Freezing the edges wastes the most informative channel, so I added an edge update each round, re-deriving every contact's meaning from the refreshed nodes. Distances are a complete, automatically-invariant local description if I use enough of them, so I replaced orientation/quaternion features with the 25 inter-atomic distances among `N, CA, C, O` and a virtual `Cβ`, each in a 16-Gaussian RBF, and zeroed the node features so all geometry enters through edges. The ablation evidence said MLP-over-neighbors beats attention here, so the layer is a three-linear message summed over neighbors and divided by a fixed `scale ≈ 30` to keep the magnitude degree-independent. Any permutation is a valid factorization, so I made decoding order-agnostic — random orders at training (a denser, BERT-like signal and design-time flexibility), implemented by conjugating the lower-triangular order-space mask through the permutation matrix and gathering to neighbors, with sequence flowing only from already-decoded neighbors. And the perfect-coordinate distribution shift demanded Gaussian backbone noise as input augmentation, a robustness knob I sweep. Three encoder layers with node+edge updates, three decoder layers, hidden 128, GELU, dropout 0.1, label smoothing 0.1, Noam-scheduled Adam, masked cross-entropy.

Here is the method as code, the slots filled in.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

NUM_AA = 21  # 20 standard amino acids + 1 unknown/non-canonical slot

def gather_edges(edges, E_idx):
    # edges (B,L,L,C) at neighbor idx (B,L,K) -> (B,L,K,C)
    nbr = E_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, nbr)


def gather_nodes(h_V, E_idx):
    # nodes (B,L,C) at neighbor idx (B,L,K) -> (B,L,K,C)
    B, L, K = E_idx.shape
    flat = E_idx.reshape(B, L * K, 1).expand(-1, -1, h_V.size(2))
    return torch.gather(h_V, 1, flat).reshape(B, L, K, -1)


def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
    return torch.cat([h_edges, gather_nodes(h_nodes, E_idx)], dim=-1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff)
        self.W_out = nn.Linear(num_ff, num_hidden)
        self.act = nn.GELU()

    def forward(self, h):
        return self.W_out(self.act(self.W_in(h)))


class PositionalEncodings(nn.Module):
    # clipped signed residue-index offset, with one extra bucket for cross-chain edges
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset, same_chain):
        d = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        d = d * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d.long(), 2 * self.max_relative_feature + 2)
        return self.linear(d_onehot.float())


class ProteinFeatures(nn.Module):
    """Edges = 25 inter-atomic distances (N,CA,C,O,virtual Cb) in Gaussian RBFs + relative position.
    Nodes carry NO geometry (initialized to zero in the encoder)."""
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16,
                 top_k=30, augment_eps=0.0):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt((dX ** 2).sum(3) + eps)
        D_max, _ = D.max(-1, keepdim=True)
        D = D + (1.0 - mask_2D) * D_max                              # push padded to the back
        D_nbr, E_idx = torch.topk(D, min(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_nbr, E_idx

    def _rbf(self, D):
        mu = torch.linspace(2.0, 22.0, self.num_rbf, device=D.device).view(1, 1, 1, -1)
        sigma = (22.0 - 2.0) / self.num_rbf
        return torch.exp(-((D.unsqueeze(-1) - mu) / sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D = torch.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1) + 1e-6)  # (B,L,L)
        D_nbr = gather_edges(D[:, :, :, None], E_idx)[:, :, :, 0]                    # (B,L,K)
        return self._rbf(D_nbr)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)          # backbone-noise augmentation

        N, Ca, C, O = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :], X[:, :, 3, :]
        # virtual Cb at the ideal tetrahedral position from backbone geometry
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        D_nbr, E_idx = self._dist(Ca, mask)                         # kNN graph over Ca

        atoms = [Ca, N, C, O, Cb]
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),            # Ca-Ca, N-N, C-C, O-O, Cb-Cb
                 (0, 1), (0, 2), (0, 3), (0, 4),                    # Ca-N, Ca-C, Ca-O, Ca-Cb
                 (1, 2), (1, 3), (1, 4), (4, 2), (4, 3), (3, 2),    # N-C, N-O, N-Cb, Cb-C, Cb-O, O-C
                 (1, 0), (2, 0), (3, 0), (4, 0),                    # reverse orientations
                 (2, 1), (3, 1), (4, 1), (2, 4), (3, 4), (2, 3)]
        RBF_all = torch.cat([self._get_rbf(atoms[i], atoms[j], E_idx) for i, j in pairs], dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        same_chain = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(same_chain[:, :, :, None], E_idx)[:, :, :, 0]
        E_pos = self.embeddings(offset.long(), E_chains)
        E = torch.cat([E_pos, RBF_all], -1)
        E = self.norm_edges(self.edge_embedding(E))
        return E, E_idx


class EncLayer(nn.Module):
    """Message-passing encoder layer: node update by MLP-over-neighbors, THEN edge update."""
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(num_hidden) for _ in range(3))
        self.drop1, self.drop2, self.drop3 = (nn.Dropout(dropout) for _ in range(3))
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)        # node message MLP
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)       # edge update MLP
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        # node update: gather [h_i, h_j, e_ij], MLP, mean over neighbors via fixed scale
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_i = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_i, h_EV], -1)
        m = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            m = mask_attend.unsqueeze(-1) * m                       # zero padded neighbors
        dh = m.sum(-2) / self.scale                                 # degree-stable mean
        h_V = self.norm1(h_V + self.drop1(dh))
        h_V = self.norm2(h_V + self.drop2(self.dense(h_V)))
        if mask is not None:
            h_V = mask.unsqueeze(-1) * h_V

        # edge update: re-derive e_ij from the refreshed node states
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_i = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_i, h_EV], -1)
        m = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.drop3(m))
        return h_V, h_E


class DecLayer(nn.Module):
    """Decoder layer: same message+FFN block as the encoder, no edge update."""
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask, mask_attend=None):
        h_i = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_i, h_E], -1)
        m = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            m = mask_attend.unsqueeze(-1) * m
        dh = m.sum(-2) / self.scale
        h_V = self.norm1(h_V + self.drop1(dh))
        h_V = self.norm2(h_V + self.drop2(self.dense(h_V)))
        if mask is not None:
            h_V = mask.unsqueeze(-1) * h_V
        return h_V


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=NUM_AA, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=NUM_AA, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super().__init__()
        self.features = ProteinFeatures(edge_features=edge_features, node_features=node_features,
                                        top_k=k_neighbors, augment_eps=augment_eps)
        self.W_e = nn.Linear(edge_features, hidden_dim)
        self.W_s = nn.Embedding(vocab, hidden_dim)                  # amino-acid embedding for the decoder
        self.encoder_layers = nn.ModuleList(
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers))
        self.decoder_layers = nn.ModuleList(
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers))
        self.W_out = nn.Linear(hidden_dim, num_letters)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(E.shape[0], E.shape[1], E.shape[-1], device=device)   # zero node init
        h_E = self.W_e(E)

        # encoder: unmasked message passing, nodes AND edges updated each layer
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # order-agnostic causal mask
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_enc = cat_neighbors_nodes(h_V, h_EX, E_idx)

        chain_M = chain_M * mask
        randn = torch.randn(chain_M.shape, device=device)
        decoding_order = torch.argsort((chain_M + 1e-4) * torch.abs(randn))      # fixed-first, else random
        L = E_idx.shape[1]
        P = F.one_hot(decoding_order, num_classes=L).float()                     # permutation matrix
        Lhat = 1.0 - torch.triu(torch.ones(L, L, device=device))                 # strictly-lower (order space)
        order_mask = torch.einsum('ij,biq,bjp->bqp', Lhat, P, P)                 # 1 iff p precedes q
        mask_attend = torch.gather(order_mask, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1D * mask_attend                                          # decoded neighbors
        mask_fw = mask_1D * (1.0 - mask_attend)                                  # undecoded neighbors

        # decoder: sequence flows only from already-decoded neighbors; structure from all
        h_EXV_enc_fw = mask_fw * h_EXV_enc
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_enc_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        return F.log_softmax(self.W_out(h_V), dim=-1)


def loss_smoothed(S, log_probs, mask, weight=0.1):
    # masked per-residue cross-entropy with label smoothing
    S_onehot = F.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = (loss * mask).sum() / 2000.0
    return loss, loss_av


class NoamOpt:
    # Transformer warmup schedule: lr = factor * d^-0.5 * min(step^-0.5, step*warmup^-1.5)
    def __init__(self, model_size, factor, warmup, optimizer, step=0):
        self.optimizer, self.factor, self.warmup, self.model_size = optimizer, factor, warmup, model_size
        self._step = step
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)

    def step(self):
        self._step += 1
        lr = self.rate()
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self._rate = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step=0):
    return NoamOpt(d_model, 2, 4000,
                   torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step)
```

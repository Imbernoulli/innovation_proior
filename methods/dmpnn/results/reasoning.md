Let me start from what actually hurts. I want to predict a property of a molecule — is it toxic, does it cross the blood-brain barrier, what's its solubility — from nothing but its structure, and I want the prediction to hold up on molecules a chemist hasn't synthesized yet, new scaffolds, the next series. There are two camps. One computes a fixed vector for the molecule — a Morgan fingerprint, a pile of RDKit descriptors — and runs a random forest or a feed-forward net on it. The other hands the molecular graph to a graph convolutional network and lets it learn its own encoding against the labels. The fixed-vector camp has a strong, general prior baked in by whoever designed the fingerprint, but it can never specialize to the property I actually care about; the learned camp can specialize but needs data and, worse, on a random train/test split it can cheat — it memorizes the scaffolds it saw in training and scores well on a test set that shares those scaffolds, then collapses on genuinely new chemistry. Two things to fix, then. I want a learned representation that can compete with both camps, and I want to know that any apparent gain is real and not memorization, which means I'd better evaluate on a split where train and test share no scaffold. Hold that evaluation thought; the architecture is the harder part.

So I'm in the learned camp, and the natural object is message passing on the molecular graph. Atoms are nodes with features x_v — atomic number, degree, charge, chirality, hydrogens, hybridization, aromaticity, mass — bonds are edges with features e_vw — single/double/triple/aromatic, conjugated, in a ring, stereo — and all of that comes straight out of RDKit. The standard recipe runs in two phases. A message phase, T rounds, where every atom gathers a message from its neighbors and updates a hidden state,

  m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw),
  h_v^{t+1} = U_t(h_v^t, m_v^{t+1}),

and then a readout that pools the final atom states into a molecule vector and predicts. M_t and U_t are learned and differentiable; the whole thing trains end to end. This is the framework basically every graph model of the moment fits into — Duvenaud's neural fingerprints take the message as a concatenation (h_w, e_vw) and update through a degree-specific matrix; Li's gated graph net uses an edge-typed linear message and a GRU update; Kearnes's weave net is the odd one that keeps edge representations around. They differ in M and U, but they all share one thing: the hidden state and the message live on the *atom*.

Let me actually trace a single message through that atom-centered update and see if it does what I want, because something nags at me. Pick an edge between atoms 1 and 2. At step t, atom 2 forms m_2 = sum over its neighbors of M(h_2, h_w, e_2w), and that sum runs over *all* neighbors of 2 — including atom 1. So part of the message atom 2 receives is the contribution from atom 1, h_1. Now atom 2 updates to h_2^{t+1}, which therefore contains a piece of h_1. Next step, atom 1 forms its message m_1 = sum over its neighbors, which includes atom 2, so atom 1 pulls in h_2^{t+1} — which contains the piece of h_1 it just sent over. So the message atom 1 sent to atom 2 looks like it comes straight back to atom 1 the next round. I want to be sure that's a real effect and not me telling a story, so let me count it on the smallest graph where it can happen: a path 0–1–2, with the update h_v^{t+1} = h_v^t + sum_{w in N(v)} h_w^t, and I'll just track *provenance* — for each atom, the multiset of which atoms' initial information is present in its state. Start: atom v knows only itself, {v}. One step: atom 1 has neighbors 0 and 2, so its state becomes {1; 0; 2}. Two steps: atom 0 (neighbor 1) now carries {0; 1} after step one and atom 2 carries {2; 1}, and atom 1 absorbs both plus its own step-one state {1;0;2}, giving — counting multiplicities — {1:3, 0:2, 2:2}. So atom 1 hears its *own* initial information three times over. Two of those copies are pure round-trips: the {1} it pushed onto atom 0 and onto atom 2 at step one, reflected straight back at step two. I'm passing a message out one bond and immediately back along the same bond. Walks that do this — out and immediately back, v_1 v_2 v_1, the pattern v_i = v_{i+2} — have a name, totters. The multiplicity-3 I just counted is them: they don't bring in new information, they re-mix information the node already had, and the count only grows with depth and degree. That's not a corner case; it's structural — it happens on every edge of every molecule on every step. This is the thing I have to kill.

And I half-recognize the disease, because I've seen the cure in a completely different context. Belief propagation on a graphical model passes messages along edges too, and there the message from node v to a neighbor w is built from everything v has heard *except* the message that came from w. That exclusion isn't a nicety — it's the whole reason BP doesn't double-count on a tree: you never tell a neighbor something it already told you. The atom-centered update violates exactly this. It sums over all of N(v) with no exclusion, so it tells atom 1 what atom 1 just said. If I want the BP-style exclusion, the update needs to know which neighbor the message is *going to* so it can leave that one out. But a hidden state sitting on an undirected atom has no notion of "going to"; it's a single vector per atom, direction-blind. So the fix can't just be "drop a term" — I need to change *where the state lives* so that direction is representable at all.

The state itself is in the wrong place. Keep a separate hidden state for the bond from v to w and for the bond from w to v — call them h_vw and h_wv, and insist they're distinct, h_vw ≠ h_wv. Now "the message heading from v out to w" is a first-class object I can name, and I can build it from the *other* directed bonds flowing into v. Concretely, the message that will become the bond v→w should aggregate the incoming bonds at v — the k→v bonds — but skip the one bond I must not feed back, which is w→v, the reverse of where I'm headed:

  m_vw^{t+1} = sum_{k in N(v)\w} h_kv^t.

That's the exclusion I want, now expressible because the state is directional: the message into edge v→w does not depend on the reverse message h_wv from the previous step. Does removing that one term actually cancel the totter, though, or just hide it? Let me rerun the provenance count, this time on directed bonds with the exclusion in place — bond v→w starts knowing its source atom v, and m_vw aggregates the incoming k→v bonds with k≠w. On the same path 0–1–2, track bond 1→2 for two steps. With the exclusion, after two steps its provenance is {1:1, 0:1} — atoms 1 and 0, the part of the graph genuinely upstream of that bond, and atom 2 appears with count *zero*. Now drop the exclusion (sum over all of N(1), reverse included) and rerun: bond 1→2 comes out {1:3, 0:1, 2:1} — atom 2's information has reflected back in (the 2→1 bond fed straight into 1→2), and atom 1's own count has ballooned to three, the very totter multiplicity I saw in the atom-centered trace. So the single excluded term is exactly the back-flow: with it gone the self-echo count drops from nonzero to zero. The fix works, and I can see it works rather than assert it.

I half-recognize this shape from belief propagation: there the message v→w is built from everything v has heard *except* what came from w, and that exclusion is precisely why BP doesn't double-count on a tree. Written as a learned fixed-point update it would be nu_ij = sigma(W1 x_i + W2 sum_{k in N(i)\j} nu_ki) — a directed-bond message summing over N(i) minus the destination, which is the form I just landed on. The node-centered version mu_i = sigma(W1 x_i + W2 sum_j mu_j) sums over *all* neighbors with no exclusion — that's the atom MPNN, the one whose provenance count blew up. So the choice between atom and directed-bond messages lines up with the choice between the mean-field-style and the loopy-BP-style update, and it's the latter that carries the exclusion. I'd want to check the correspondence is more than cosmetic before leaning on it, but as a guide for *where the state should live* it points the same way the totter count did.

Now I have to make the rest of the update concrete and decide what M and U are. Let me keep the message function trivial: M_t just returns the bond hidden state, so the "message" sent along k→v is literally h_kv^t. All the learning goes into the update U_t. Before the first message round I have to initialize each directed bond from its raw features. The bond v→w should start from the atom it leaves, v, plus the bond's own features — concatenate x_v with e_vw and push through a learned matrix and a nonlinearity:

  h_vw^0 = tau(W_i cat(x_v, e_vw)),

with tau a ReLU. Note this already does something Duvenaud's concatenation message couldn't: by mixing x_v and e_vw together inside one matrix W_i, the initial bond state can represent correlations between the source atom and the bond, instead of summing the two streams separately and losing their interaction. Then each round I take the aggregated incoming message and update. The simplest update is a learned linear map of the message through a shared matrix W_m, then a nonlinearity:

  h_vw^{t+1} = tau(W_m m_vw^{t+1}).

But let me think about depth. I'm going to unroll this for T steps with the *same* W_m every step — weight tying, like the GRU in the gated graph net, which keeps the parameter count flat regardless of how many rounds I run and makes the depth a pure hyperparameter rather than a stack of distinct layers. The risk with a tied recurrent update is that after a few rounds the bond's original identity gets washed out — each round overwrites the state, and the raw bond features that h_vw^0 captured fade. I don't want that; the bond's own features (it's a double bond in a ring, conjugated) are useful at *every* depth, not just step zero. So re-inject them: add h_vw^0 back at every step before the nonlinearity,

  h_vw^{t+1} = tau(h_vw^0 + W_m m_vw^{t+1}).

That's a skip connection straight to the original bond features. It keeps the gradient path short through the tied recurrence, and because h_vw^0 is added back at every single step, the raw bond features are re-presented to the state no matter how deep I go rather than having to survive T rounds of being overwritten. Cheap, and it directly addresses the wash-out failure mode of a tied update.

After T rounds I have rich directed-bond states but I need to predict a molecule property, which lives at the level of the whole molecule, so I have to come back to atoms and then to one vector. Returning to atoms is the natural aggregation: an atom's representation is the sum of the bond messages flowing *into* it,

  m_v = sum_{w in N(v)} h_wv^T,

and then combine that with the atom's own features — because the bonds-only messages may have drifted from what the atom intrinsically is, so re-inject x_v the same way I re-injected bond features —

  h_v = tau(W_a cat(x_v, m_v)).

And the molecule vector is the sum over atoms,

  h = sum_{v in G} h_v,

which is permutation-invariant — it doesn't care about the arbitrary order I numbered the atoms in, which I'd better respect since a molecule has no canonical atom ordering. Then a feed-forward net f maps h to the prediction, y_hat = f(h). One readout subtlety I should at least flag: summing over atoms makes h grow with molecule size, which can be fine or can be a nuisance depending on the property; a mean would normalize it out. I'll take the sum as the default and keep mean as a knob.

Now I want to be honest about cost, because the exclusion "sum over N(v)\w" looks like it could be expensive. Naively, for every directed bond v→w I sum over the neighbors of v except w, and if I literally loop that, it's O(number of bonds times average degree) per step, and I'm doing it on every bond of every molecule in every batch. That's wasteful, and there's a clean trick. The sum over N(v)\w is just the sum over *all* of N(v) minus the one term I'm excluding. The sum over all incoming bonds at v doesn't depend on the destination w at all — I can compute it once per atom:

  a_v = sum_{k in N(v)} h_kv^t,   one aggregation per atom, O(bonds total),

and then for each outgoing bond v→w the message I want is that atom-level sum minus the single reverse bond's contribution:

  m_vw^{t+1} = a_v - h_wv^t.

So the per-bond cost is one subtraction. The exclusion went from "loop over neighbors each time" to "aggregate once per atom, then subtract the reverse." But I should check that this rewrite is actually an identity and not just plausible — it's the kind of thing that's off by exactly the term you forgot. Take the path 0–1–2 again, directed bonds stored as 0→1, 1→0, 1→2, 2→1, and give them arbitrary distinct states, say 10, 20, 30, 40. Aggregate incoming per atom: atom 1 receives the 0→1 bond (10) and the 2→1 bond (40), so a_1 = 50. The message into bond 1→2 should be sum over N(1)\{2}, which is just the 0→1 bond, = 10. The formula gives a_1 minus the reverse of 1→2, which is the 2→1 bond (40): 50 − 40 = 10. It matches. I ran the full vectorized version against a brute-force loop over N(v)\w for every directed bond and they agree element-for-element, so a[src] − h[rev] is the exclusion, not an approximation to it.

For this to be O(1) per bond I also need to find the reverse bond w→v of each bond v→w instantly. If I store bonds in pairs — every undirected bond becomes two consecutive entries, v→w at some index e and w→v at e+1 — then the reverse of bond e should be e with its lowest bit flipped, e XOR 1. Let me confirm the bit trick lands where I claim: with bonds [0→1, 1→0, 1→2, 2→1] at indices 0,1,2,3, the XOR-1 map sends 0↔1 and 2↔3, and indeed bond 0 (0→1) pairs with bond 1 (1→0), bond 2 (1→2) with bond 3 (2→1) — every bond mapped to its true reverse, no search. (I verified src[rev]==dst and dst[rev]==src hold for all bonds, which is exactly the assertion the code guards on.) So directed-bond message passing with the exclusion costs essentially the same as a plain atom aggregation, because "sum over all then subtract the one" is linear in the bonds and the reverse lookup is a single XOR. The architectural change didn't cost me anything computationally.

Let me write the message phase as the vectorized loop I'd actually run, because I want it droppable into the graph-batch harness. I lay out a batch as one big disjoint graph: a stacked atom-feature matrix, a stacked bond-feature matrix, and index arrays — for each bond its source and destination atom, for each atom the list of its incoming bonds, and the reverse-bond index. Then:

  input  = W_i(cat over bonds of (x_src, e_bond));  h^0 = ReLU(input)
  message = h^0
  repeat T-1 times:
    a = scatter-sum the bond messages into their destination atoms     # a_v = sum_{k in N(v)} h_kv
    rev = message[reverse_index]                                        # h_wv for each bond v->w
    message = a[source_atom_of_each_bond] - rev                         # m_vw = a_v - h_wv  (the N(v)\w sum)
    message = ReLU(h^0 + W_m(message))                                  # tied update + skip to h^0
  # back to atoms:
  a = scatter-sum bond messages into destination atoms                 # m_v = sum_{w in N(v)} h_wv^T
  h_atom = ReLU(W_a(cat(x_atom, a)))                                    # h_v = tau(W_a cat(x_v, m_v))
  h_mol  = scatter-sum h_atom over each molecule's atoms                # h = sum_v h_v

The loop runs T-1 times for the updates and then one more aggregation closes it out into atoms — that's T message rounds total in the convention where h^0 is the zeroth. Everything is scatter-sums, gathers, two matmuls, a subtraction, a ReLU. The subtraction `a[src] - rev` is the directed exclusion; the `ReLU(h^0 + ...)` is the tied update with the skip. This is the entire encoder.

So far this is a pure learned representation, and I want it to compete with the fixed-descriptor camp. But I should be suspicious of my own model in two specific regimes, and both fall out of the structure of the message passing — I can see them by reasoning about the architecture, before measuring anything. One is depth. I run T rounds — three, say — and T is almost always smaller than the diameter of a real drug-like molecule, which can be a dozen bonds across. An atom only ever hears about atoms within T bonds of it, so the representation it builds is fundamentally *local*. Anything genuinely global about the molecule — a property that depends on two distant substructures jointly — simply cannot reach a single atom's state in T hops. The pooled molecule vector is a sum of local views, and it can miss global structure. The other is data. Many of these datasets are tiny, hundreds to a couple thousand molecules. A model learning its entire representation from scratch has almost nothing to learn from; it overfits the artifacts of a small training set, and in the lowest-data regime the fixed-descriptor models, which carry a strong external prior, can simply outperform it. Both failures point the same way: the learned encoding is local and data-hungry, and I have a cheap external source of global, prior chemical knowledge sitting right there — the very RDKit molecule-level descriptors the other camp uses.

So bring them in when I have them, but as a *complement* rather than a replacement. Compute a few hundred fast molecule-level descriptors h_f with RDKit, and instead of feeding the FFN only the learned vector h, feed it the concatenation:

  y_hat = f(cat(h, h_f)).

This is a hybrid: the learned message-passed h supplies task-specific, locally-resolved structure; the fixed h_f supplies a global, general chemical prior that doesn't need T to be large and doesn't need much data to be useful. On a small dataset h_f regularizes — it hands the model a view of a much larger chemical domain it could never have learned from a few hundred molecules. On a property that's global, h_f reaches across the molecule where h can't. It's a very general trick — any computed descriptor can ride along into any message passing readout — and it directly patches the two failure modes I reasoned out from the architecture. The core encoder is still the same graph-to-vector map; the descriptor branch is an extra readout input when those descriptors are supplied.

There's a normalization wrinkle I can't ignore, though. These descriptors have wildly different scales — a molecular weight in the hundreds next to a fraction between zero and one next to an integer ring count — and if I dump raw values into the FFN, the large-range features dominate and the small ones vanish. The obvious fix is min-max or z-score scaling, but think about what these features *are*. Min-max is wrecked by a single outlier molecule that stretches the range. Z-score assumes the feature is roughly normal, which is false for the count-based chemical features — number of azide groups is mostly zero with a rare spike, nothing like a Gaussian. What I actually want is a scaling where every feature carries the *same meaning* and is robust to both outliers and weird distributions. Map each raw value through the feature's cumulative distribution function: the transformed value is "the fraction of molecules with a smaller raw value," a percentile in [0,1]. That's identical in meaning across every feature, immune to outliers (an extreme value just maps near 1), and makes no normality assumption. Fit the CDFs once on a large background sample of molecules so the transform is stable and the same on train and test, then apply it. In a streaming training setting I'd approximate the same standardization with running per-feature statistics updated as batches arrive — a batch-norm-style normalizer on the descriptor branch — which gets me the scale-equalization without a precomputed CDF table. Either way the point is the same: standardize the descriptor branch before it meets the FFN.

Now the few remaining knobs, and I want each to follow from something, not be pulled from a hat. The activation is ReLU throughout — cheap, nonsaturating, and h^0 is born through it so the states stay nonnegative and gradients flow. The aggregation is a sum for permutation invariance, with mean as the size-normalizing alternative. Depth T around three: enough hops to build a useful local environment around each atom — three bonds reaches a respectable neighborhood — without so many that the tied recurrence either over-smooths everything toward a uniform blob or, even with the skip connection, starts to wash out distinctions; and the cost is linear in T so I'm not paying much either way. Hidden size a few hundred, say 300, the usual sweet spot for these molecule sizes. The FFN is a small two-layer head — one hidden layer with a nonlinearity and dropout, then a linear projection to the number of tasks; for multi-task datasets it outputs one logit per task at once. Dropout I'll keep at zero by default and turn up per dataset when overfitting bites, especially on the small ones. Training is Adam with a warmup-then-decay learning-rate schedule — ramp the rate up linearly for the first couple of epochs so the early, badly-initialized steps don't blow up, then decay it exponentially toward the end so I settle — and Xavier-normal initialization on the matrices with zeros on the biases. For multi-task classification with missing labels I use a binary-cross-entropy loss masked to the labels that are actually present, so absent assays contribute no gradient; for regression I normalize the targets first and use a squared error.

Let me close the loop back on the evaluation worry I parked at the start, because it's part of the method, not an afterthought. If I test on a random split, my learned model can score high by recognizing scaffolds it memorized, and I'd fool myself. So I split by scaffold: compute each molecule's Murcko scaffold, and partition so that no scaffold appears in both train and test. Now the test molecules are structurally novel, which is the only honest measure of generalization to new chemistry and the closest public-data stand-in for the chronological split a real discovery pipeline lives or dies by. The architecture and the evaluation are answering the same question — does the representation transfer — from two ends.

Putting the whole chain together: the atom-centered message passing everyone uses tots, because summing over all neighbors sends each message straight back where it came from, polluting the representation with its own echoes every step. The cure is the belief-propagation exclusion — build the message into an edge from the *other* edges, not the reverse one — but an undirected atom state can't even express "the reverse one," so I move the state onto directed bonds, where m_vw = sum_{k in N(v)\w} h_kv is exactly the loopy-BP embedding and the reverse message drops out by construction. I make the update a tied linear map with a ReLU and a skip connection back to the initial bond features h^0, so depth is free and the bond's raw identity never washes out. The exclusion is computed as "sum over all incoming bonds at the source atom, then subtract the single reverse bond," which is linear in the bonds with an XOR-1 reverse lookup, so the directed scheme costs no more than a plain aggregation. I return to atoms by summing the directed bonds that end at each atom and re-injecting atom features, then sum over atoms for a permutation-invariant molecule vector. Because the encoding is local (T < diameter) and data-hungry on small sets, CDF-normalized RDKit descriptors can be concatenated as a global, prior-laden complement before the FFN whenever the data pipeline supplies them. And I evaluate on a scaffold split so any improvement measures transfer to new chemistry rather than scaffold memorization. Here is the graph encoder and head for the fixed harness; the descriptor branch is the same concatenation step when a batch supplies molecule-level features:

```python
import torch
import torch.nn as nn


def scatter_sum(src, index, dim_size):
    """Sum rows of `src` into `dim_size` buckets given by `index`."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden_dim, self.depth = hidden_dim, depth
        self.W_i = nn.Linear(atom_dim + edge_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        n_atoms, n_bonds = x.size(0), edge_index.size(1)
        if n_bonds == 0:
            h_atom = self.act(self.W_a(torch.cat(
                [x, torch.zeros(n_atoms, self.hidden_dim, device=x.device, dtype=x.dtype)], dim=-1)))
            return self.dropout(h_atom)

        if n_bonds % 2 != 0:
            raise ValueError("Directed bonds must be stored as adjacent forward/reverse pairs.")
        rev = torch.arange(n_bonds, device=x.device) ^ 1
        if not bool(((src[rev] == dst) & (dst[rev] == src)).all().item()):
            raise ValueError("edge_index must store each reverse bond at index e XOR 1.")

        h0 = self.act(self.W_i(torch.cat([x[src], edge_attr], dim=-1)))
        h = h0
        for _ in range(self.depth - 1):
            a = scatter_sum(h, dst, n_atoms)         # a_v = sum_{k in N(v)} h_kv
            m = a[src] - h[rev]                      # m_vw = a_v - h_wv
            h = self.act(h0 + self.W_m(m))
            h = self.dropout(h)

        m_v = scatter_sum(h, dst, n_atoms)           # m_v = sum_{w in N(v)} h_wv^T
        h_atom = self.act(self.W_a(torch.cat([x, m_v], dim=-1)))
        return self.dropout(h_atom)


class MoleculeModel(nn.Module):
    def __init__(self, atom_dim, edge_dim, num_tasks, task_type):
        super().__init__()
        self.num_tasks, self.task_type = num_tasks, task_type
        hidden_dim, depth = 300, 3
        dropout = 0.0
        self.encoder = DMPNNEncoder(atom_dim, edge_dim, hidden_dim, depth, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_tasks))

    def forward(self, batch):
        h_atom = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        n_mol = int(batch.batch_idx.max().item()) + 1
        h = scatter_sum(h_atom, batch.batch_idx, n_mol)
        return self.head(h)
```

The encoder is the directed-bond message passing with the exclusion done as sum-minus-reverse and the tied-update-plus-skip; the model wraps it with the sum readout, and the head is the small FFN that turns the molecular vector into one prediction per task. When the feature branch is available, I standardize the RDKit descriptor vector, concatenate it to `h` before the first head layer, and widen that first layer from `hidden_dim` to `hidden_dim + feature_dim`.

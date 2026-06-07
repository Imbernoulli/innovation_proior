# MPNN synthesis (pre-Phase-2)

## Pain point / research question
Predict quantum-mechanical properties of small organic molecules (QM9: 134k molecules, ≤9 heavy atoms C/N/O/F + H; 13 regression targets approximated by DFT). DFT is accurate but slow (~1h/molecule for 9 heavy atoms, O(N_e^3)). Want a learned surrogate that is ~10^5 faster, learns its own features from the molecular graph directly, and is invariant to graph isomorphism (atom relabeling). Target: "chemical accuracy" per-target MAE thresholds.

## Field state at the time
- Chemistry ML dominated by hand-engineered molecular representations + off-the-shelf regressor: Coulomb Matrix (Rupp 2012), Bag of Bonds (BoB), BAML, ECFP4 fingerprints (Rogers&Hahn 2010), HDAD projected histograms (Faber 2017). These build symmetries into the input by hand; CM is not isomorphism-invariant (must be learned via augmentation); Behler-Parrinello symmetry functions invariant but fail >3 atom species / novel compositions.
- A cluster of neural graph models had appeared, all sharing a structure: each node holds a hidden state, repeatedly aggregates info from neighbors, then a graph-level pooling produces output. But each was presented in its own notation, making them look unrelated.

## The unifying abstraction (the contribution to derive)
Forward pass = message passing phase (T steps) + readout phase.
- message:  m_v^{t+1} = Σ_{w∈N(v)} M_t(h_v^t, h_w^t, e_{vw})       (eq 1)
- update:   h_v^{t+1} = U_t(h_v^t, m_v^{t+1})                       (eq 2)
- readout:  ŷ = R({ h_v^T | v∈G })   with R permutation-invariant   (eq 3)
M_t, U_t, R learned differentiable. Permutation invariance of R + the symmetric sum over N(v) ⇒ whole model invariant to graph isomorphism. (Could also carry edge states h_{e_{vw}}^t updated analogously; only Kearnes does.)

### Instantiations (each prior model = a choice of M,U,R)
1. Duvenaud 2015 (molecular fingerprints): M(h_v,h_w,e_{vw})=(h_w,e_{vw}) [concat]; U_t=σ(H_t^{deg(v)} m_v^{t+1}) — separate learned matrix per (timestep, degree); R = f(Σ_{v,t} softmax(W_t h_v^t)) with skip connections to all t. Weakness: message = (Σh_w, Σe_{vw}) sums nodes and edges separately ⇒ can't see node–edge correlations.
2. GG-NN (Li/Yujia 2016): M_t = A_{e_{vw}} h_w^t (one learned matrix per discrete edge label); U_t = GRU(h_v^t, m_v^{t+1}), weight-tied across t; R = Σ_v σ(i(h_v^T,h_v^0)) ⊙ j(h_v^T) (gated sum, eq for graph_level). This is the strong baseline they build on.
3. Interaction Networks (Battaglia 2016): M = NN(concat(h_v,h_w,e_{vw})); U = NN(concat(h_v,x_v,m_v)) with external x_v; graph-level R = f(Σ_v h_v^T). Originally only T=1.
4. Molecular graph convolutions (Kearnes 2016): node msg M=e_{vw}^t (edge state itself); U_t=α(W_1(α(W_0 h_v^t), m_v^{t+1})); also updates edge states e_{vw}^{t+1}=α(W_4(α(W_2 e_{vw}^t), α(W_3(h_v^t,h_w^t)))). α=ReLU. Only model with edge states.
5. Deep Tensor NN (Schütt 2017): M_t=tanh(W^{fc}((W^{cf}h_w^t+b1)⊙(W^{df}e_{vw}+b2))); U_t=h_v^t+m_v^{t+1} (residual); R=Σ_v NN(h_v^T).
6. Laplacian/spectral (Bruna 2013, Defferrard 2016, Kipf&Welling 2016): M_t=C_{vw}^t h_w^t (C from Laplacian eigvecs); U=σ(m). Kipf: M=c_{vw}h_w with c_{vw}=(deg v · deg w)^{-1/2} A_{vw} (scalar), U=ReLU(W^t m). Spectral GCN→MPNN derivation in appendix (below).

### Appendix derivation — spectral GCN as MPNN (must be lived out in reasoning)
Bruna eq (3.2): y_j = σ( Σ_{i=1}^{d1} V F_{ij} V^T x_i ), j=1..d2. V eigvecs of L=I - D^{-1/2}WD^{-1/2}, F_{ij} diagonal NxN learned.
Define rank-4 tensor L̃_{v,w,i,j}=(V F_{ij} V^T)_{v,w}. Then
  y_{v,j} = σ(Σ_{i,w} L̃_{v,w,i,j} x_{w,i})  ⇒  y_v = σ(Σ_w L̃_{v,w} x_w),  L̃_{v,w} a d1×d2 matrix.
Relabel y_v→h_v^{t+1}, x_w→h_w^t: M(h_v,h_w)=L̃_{v,w} h_w, U(h,m)=σ(m). QED.
Kipf special case: H^{l+1}=σ(D̃^{-1/2}Ã D̃^{-1/2} H^l W^l), Ã=A+I (self loops), D̃_ii=Σ_j Ã_ij.
Let L=D̃^{-1/2}ÃD̃^{-1/2}. Row v: H^{l+1}_{(v)}=σ(Σ_w L_{vw} H^l_{(w)} W^l) ⇒ h_v^{t+1}=σ((W^l)^T Σ_w L_{vw} h_w^t).
So M_t(h_v,h_w)=L_{vw}h_w = Ã_{vw}(deg v·deg w)^{-1/2} h_w (scalar weight), U_t(h,m)=σ((W^t)^T m). Weighted average of neighbors.

## The QM9-specific design choices (derive each WHY)
Built around GG-NN as base. d = node hidden dim, n = #nodes.
- Directed treatment: separate M^in, M^out channels; undirected edge → in+out edge same label; message channel size 2d. Reason: lets parameter tying distinguish information flow direction; matches GG-NN family.
- Init: h_v^0 = atom feature vector x_v padded to d. Weight tying across t + GRU update (as GG-NN). Found: tie weights + bigger d beats untied weights.
- Message function options:
  - Matrix multiply (GG-NN): M=A_{e_{vw}} h_w, needs DISCRETE edge labels.
  - **Edge network (the contribution)**: M=A(e_{vw}) h_w where A(·) is an MLP mapping the edge VECTOR e_{vw} → a d×d matrix. Reason: bonds carry continuous info (spatial distance); discrete labels throw that away. Edge network handles vector-valued edges. This is the winning enn.
  - Pair message (from Interaction Nets): m_{wv}=f(h_w,h_v,e_{vw}), message depends on BOTH endpoints. Reason to try: matrix-mult message ignores h_v (destination). Outcome: trained worse than edge network (1.53 vs 3.98 avg joint), even on a toy pathfinding task designed to favor it; harder to train ⇒ dropped.
- Virtual graph elements (to carry long-range info when no spatial input):
  - Virtual edge: add a distinct edge type for every non-bonded pair (preprocessing) → info travels far in one step.
  - Master node: a latent node connected to all nodes by a special edge type, own dim d_master, own GRU; global scratchpad. Complexity O(|E|d^2 + n·d_master^2) — extra capacity without quadratic blowup.
- Readout:
  - GG-NN gated readout (eq graph_level).
  - **set2set (Vinyals 2015)**: order-invariant set encoding via M steps of LSTM-attention over the projected tuples {(h_v^T, x_v)}, outputs q*_t; feed through NN. Reason: more expressive than a plain sum; designed for sets so stays permutation-invariant. Winning readout.
- Towers (efficiency): split d-dim h_v into k copies of dim d/k, run propagation on each copy with separate M,U, then mix: (h_v^{t,1..k}) = g(h̃_v^{t,1..k}) with g a shared NN over concatenation. A matmul prop step: per copy O(n^2 (d/k)^2), k copies ⇒ O(n^2 d^2/k) total vs O(n^2 d^2). k=8,n=9,d=200 ⇒ ~2x faster. Mixing preserves permutation invariance; resembles ensembling (helps generalization). Didn't combine well with edge network.
- Input representation: atom features (type one-hot H/C/N/O/F, atomic number, acceptor, donor, aromatic, hybridization sp/sp2/sp3, #H). Edges: (a) chemical graph = discrete bond type {single,double,triple,aromatic}; (b) distance bins = 14-symbol alphabet (bond type for bonded; distance bin for non-bonded; [2,6] into 8 bins + [0,2] + [6,∞]); (c) raw distance = 5-dim edge vector [euclidean dist, one-hot bond type] for the edge network. Found: full edge vector (bond+distance) + explicit hydrogens important (graphs up to 29 nodes, ~10x slower).
- Training: per-target individual model > joint (up to 40% better). Adam, batch 20, 3M steps, lr in [1e-5,5e-4] with linear decay. Normalize targets mean0/var1. Minimize MSE, evaluate MAE. T in [3,8] (any T≥3 fine), set2set M in [1,12].

## Canonical code grounding (code/)
PyG: NNConv = edge network conv (x'_i = Θ x_i + Σ_j x_j · h_Θ(e_{ij}), h_Θ MLP edge→in*out matrix). Set2Set = the readout (LSTM attention, out dim 2·in). qm9_nn_conv.py = full enn-s2s: lin0 pad x→dim; one NNConv with nn=MLP(5→128→dim*dim) shared across T=3 steps, GRU update; Set2Set(processing_steps=3); lin1(2dim→dim)→lin2(dim→1); Adam; MSE train / MAE eval; complete graph + T.Distance edge feature. This is the faithful skeleton for answer.md / reasoning.md final code.

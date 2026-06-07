# Graphormer synthesis notes

## Pain point / research question
Transformers dominate sequences & vision, but on graph-level prediction leaderboards (OGB, Benchmarking-GNN), message-passing GNNs still win; the only "Transformer for graphs" that worked was really a GNN with softmax replacing the aggregator. Question: can a *standard* Transformer encoder model graphs well? Core obstacle: self-attention is permutation-equivariant and computes only semantic similarity QK^T — it sees a graph as a bag of node feature vectors with NO structure (no degree/importance, no distance, no edges). Sequences fix the "set" problem with positional encodings; graphs have no canonical order, but they DO have degree, shortest-path distance, and edge features. So: how to feed structure into attention without abandoning the standard Transformer?

## The three encodings (derive inline)
1. **Centrality encoding.** QK^T weights nodes by feature similarity only; node importance (degree centrality) is ignored. Add learnable embeddings indexed by in/out-degree to the input node features:
   h_i^(0) = x_i + z^-_{deg^-(v_i)} + z^+_{deg^+(v_i)}, z^-,z^+ ∈ R^d. (Undirected: single deg.) Now Q and K carry degree info, so attention can condition on importance. (Code: GraphNodeFeature, in_degree_encoder/out_degree_encoder nn.Embedding added to atom_encoder sum.)
2. **Spatial encoding.** Attention's global receptive field needs a notion of position/locality. Sequences use absolute/relative PE; graphs are not a sequence. Define φ(v_i,v_j) = shortest-path distance (SPD) (−1 / special if disconnected). Assign each SPD value a learnable scalar bias added to the QK score:
   A_ij = (h_i W_Q)(h_j W_K)^T / sqrt(d) + b_{φ(i,j)}. Shared across layers, one scalar per distance per head. If b decreasing in φ → local attention; learnable → adaptive. (Code: spatial_pos_encoder nn.Embedding(num_spatial, num_heads); SPD via Floyd-Warshall in algos.floyd_warshall.)
3. **Edge encoding.** Edges have features (bond type). Prior methods: add edge feat to node feat, or use in aggregation — both only reach the edge's endpoints. Instead, for node pair (i,j), take a shortest path SP_ij = (e_1..e_N), and average dot-products of each edge feature with a per-position learnable weight, add as bias:
   c_ij = (1/N) Σ_{n=1}^N x_{e_n} (w^E_n)^T ; A_ij += c_ij. (Code: edge_encoder, multi_hop path edge features via gen_edge_input, averaged over path positions, edge_dis_encoder for per-distance weight.)

## Architecture details
- Pre-LN Transformer (LN before MHA and FFN) — better optimization (Xiong 2020).
  h'^(l) = MHA(LN(h^(l-1))) + h^(l-1); h^(l) = FFN(LN(h'^(l))) + h'^(l).
- FFN inner dim set to d (not 4d) — saves params, no real perf loss.
- [VNode]: special virtual node connected to all nodes; its final-layer representation is h_G (graph readout). Like BERT [CLS]. Distance φ([VNode],·)=1 but connection is virtual → reset b_{φ(VNode,·)} to a distinct learnable scalar (distinguish virtual vs physical). (Code: graph_token embedding prepended; graph_token_virtual_distance added to row/col 0 of attn bias.)

## Expressiveness / GNN-subsumption (Appendix A, prove inline)
Fact 1: Graphormer layer can represent AGGREGATE+COMBINE of GIN, GCN, GraphSAGE.
- **MEAN AGGREGATE**: in A_ij = QK/√d + b_φ, set b_φ=0 if φ=1, −∞ otherwise; W_Q=W_K=0 (so QK part=0, all scores equal within 1-hop neighbors), W_V=I. Then softmax(A)V = uniform average over the 1-hop neighbors = MEAN.
- **SUM AGGREGATE**: SUM = degree × MEAN. Extract degree from centrality encoding via an extra head, concat to mean-aggregated rep; FFN multiplies mean by degree (universal approx of FFN) → SUM.
- **MAX AGGREGATE**: per dim t, one head: b_φ=0 if φ=1 else −∞; W_K=e_t, W_Q=0 with Q bias = T·1 (T large temperature), W_V=e_t → softmax→ hard max over neighbors at dim t.
- **COMBINE**: extra head that returns the node's own feature: b_φ=0 if φ=0 (self) else −∞; W_Q=W_K=0, W_V=I → returns h_i. AGGREGATE head + this self-head + FFN approximate any COMBINE.
So GCN (mean), GraphSAGE (max), GIN (sum + MLP combine) are special cases.
Fact 2 (VNode/READOUT): a vanilla Graphormer layer (no extra encodings) can represent MEAN READOUT: W_Q=W_K=0, Q,K bias = T·1 with T ≫ scale of b_φ so T^2·1·1^T dominates → all attention scores equal → softmax uniform over ALL nodes; W_V=I → mean over whole graph. So self-attention naturally does graph-level readout; this justifies/explains the VNode.
Beyond 1-WL: SPD sets distinguish graph pairs that 1-WL cannot (Fig: two graphs, identical 1-WL colors but different multisets of SPD-to-others). So spatial encoding pushes expressiveness past 1-WL ceiling of message-passing GNNs.

## Baselines / lineage
- Vanilla Transformer (Vaswani 2017): self-attn QK^T/√d softmax V, FFN, residual+LN; permutation-equivariant, needs PE for order. Structure-blind on graphs.
- Message-passing GNN framework (AGGREGATE-COMBINE-READOUT). GCN (Kipf 2017, mean/symmetric-normalized agg + linear+ReLU), GraphSAGE (Hamilton 2017, sampled neighbors, max/mean/LSTM agg, concat combine), GIN (Xu 2019, sum agg + MLP, 1-WL-expressive). Limitation: 1-hop locality, receptive field grows only with depth → over-smoothing; bounded by 1-WL; edge features handled crudely.
- VNode/supernode trick (Gilmer 2017 MPNN, OGB leaderboard): add supernode connected to all → graph-level aggregation; but naive supernode → over-smoothing.
- Relative position bias in sequences (Shaw 2018, T5 / Raffel 2019): add learned scalar to attention by relative offset — the template spatial encoding generalizes to SPD on graphs.
- Prior graph-Transformers: GT (Dwivedi 2021) Laplacian-eigvec PE + neighbor-masked attention; SAN; Graph-BERT (WL/hop PE). Limitation: restrict attention to neighbors (sparsity) or use spectral PE; don't keep full Transformer + simple structural biases.

## Evaluation settings (no outcomes)
Datasets: PCQM4M-LSC (OGB-LSC, 3.8M molecular graphs, HOMO-LUMO gap regression, MAE), OGBG-MolPCBA (AP), OGBG-MolHIV (AUC), ZINC (regression MAE). Scaffold split (OGB), uniform split (ZINC). AdamW, warmup+linear decay, pre-LN. FLAG augmentation for small datasets. Metrics: MAE / AP / ROC-AUC.

## Code structure (canonical: microsoft/Graphormer)
- preprocess_item (wrapper.py): build adj, attn_edge_type [N,N,d_e]; Floyd-Warshall → spatial_pos (SPD) + path; gen_edge_input → edge features along SP; in/out degree; attn_bias [N+1,N+1].
- GraphNodeFeature (graphormer_layers.py): atom_encoder sum + in/out_degree_encoder + prepend graph_token → [B, N+1, d].
- GraphAttnBias: spatial_pos_encoder(spatial_pos) per-head; reset row/col 0 with graph_token_virtual_distance; edge_encoder(edge along path).mean → edge bias; sum into [B, H, N+1, N+1].
- MultiheadAttention: attn_weights = QK^T*scaling; attn_weights += attn_bias; softmax; ·V.
- GraphormerGraphEncoderLayer: pre-LN MHA + FFN with residuals.
- Readout: graph_rep = x[0] (the VNode/graph token at position 0).

# Gradient checkpointing / sublinear memory synthesis (grounded)

## Verified
- arXiv 1604.06174, "Training Deep Nets with Sublinear Memory Cost", Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin.
- Canonical impl: PyTorch torch.utils.checkpoint (checkpoint, checkpoint_sequential); also openai/gradient-checkpointing (cybertronai, TF). The recompute-from-segment-boundary idea.
- Prior name: "gradient checkpointing" from automatic-differentiation literature (Griewank). Computation-graph liveness analysis from compiler register allocation (Aho dragon book).

## Pain point
- Training an n-layer net stores the intermediate feature maps (activations) of every layer for the backward pass, because most gradient operators depend on the forward intermediate results. So activation memory is O(n), linear in depth (or in sequence length n for an RNN unrolled n steps).
- GPU memory caps how deep/wide a model you can train. SOTA models (e.g. very deep ResNets) hit the memory ceiling. Parameters are small relative to activations in CNNs/RNNs; the activations dominate.
- Goal: reduce activation memory below O(n) while keeping the EXACT same gradients (no approximation), paying only a little extra compute.

## Background tools (already exist)
- Computation graph: nodes = ops, edges = dependencies. Backward pathway built by reverse-topological traversal; training = forward pass over the whole graph including the gradient pathway.
- Two memory optimizations existing frameworks (Theano/TF/MXNet) already do:
  - Inplace operation: write output into the input's memory (when input not needed elsewhere).
  - Memory sharing: recycle memory of intermediates no longer needed (liveness analysis).
- Liveness/sharing via O(n) heuristic (counter per node = number of pending consumers; recycle when counter hits 0) instead of O(n^2) graph-coloring of the conflict graph.
- These reduce PREDICTION memory from O(n) to ~O(1), but TRAINING only by a constant factor, because gradient ops still need the forward intermediates -> still O(n).

## Core idea (the trade: computation for memory)
- DROP some forward intermediates instead of storing them; RECOMPUTE them during backprop by re-running forward from the nearest stored result.
- Linear-chain version (Alg 1): split the n-layer net into segments. Store ONLY the input/output at each segment boundary, drop everything inside a segment. During backward, for each segment (in reverse): reload the segment's stored input, re-run forward through the segment to regenerate its internal activations (into a local temp), then run backward through the segment.
- Memory needed = (store segment boundaries) + (max memory to backprop within ONE segment).

## The O(sqrt(n)) derivation (Section 4.3) — the heart
- Divide n layers into k segments.
  cost-total = max_i cost-of-segment(i) + O(k) = O(n/k) + O(k)
  - O(n/k): peak memory to recompute + backprop within one segment (equal segments => n/k layers each).
  - O(k): memory to store the k segment-boundary outputs.
- Minimize O(n/k)+O(k) over k. Calculus: d/dk (n/k + k) = -n/k^2 + 1 = 0 => k = sqrt(n). Then cost = n/sqrt(n) + sqrt(n) = 2 sqrt(n) = O(sqrt(n)).
- Compute cost: backward must re-run the forward within each segment once => ONE extra full forward pass over the network (each layer's forward computed at most twice total). Since backward is ~2x forward cost, total slowdown is small (~30% measured).

## Recursion -> O(log n) (Section 4.4)
- Treat each segment as a bulk operator; recursively apply the same drop-and-recompute INSIDE the segment.
- g(n) = memory for forward+backward on n-layer net, storing k intermediates and recursing on each sub-path of length n/(k+1):
  g(n) = k + g(n/(k+1))
  Solving: g(n) = k log_{k+1}(n).
  - k=1: g(n) = log_2 n memory, at cost O(log_2 n) extra forward passes. Extreme tradeoff.

## General graph (Algs 2,3) — beyond linear chains
- Mirror count m(v): how many times node v is duplicated (re-computed). m(v)=0 => keep output (don't drop, segment boundary); m(v)=1 => recompute once (inside segment); >1 => recursion.
- Alg 2 (Memory Optimized Gradient Graph Construction): given m, builds a new graph where dropped nodes are "mirrored" (duplicated) in the backward region so they're recomputed; outputs an execution order so static allocation can optimize memory. m all 0 => normal gradient graph.
- Alg 3 (Memory Planning with Budget B): greedy single-pass over topological order; accumulate output sizes into temp; when at a candidate split point and temp>B, set m(v)=0 (keep boundary), reset temp; else m(v)=1 (drop). Search over B (grid around B=sqrt(x*y), x=inter-stage cost, y=per-stage cost) to balance the two memory terms when layer costs are non-uniform.

## Cheap special case (Section 4.2)
- Drop results of LOW-COST ops, keep expensive ones. In Conv-BatchNorm-Activation pipelines: keep the conv output, drop BN/activation/pooling outputs (cheap to recompute). Memory saving with tiny compute overhead.

## Canonical code grounding (PyTorch torch.utils.checkpoint)
- checkpoint(function, *args, use_reentrant): forward runs `function` WITHOUT saving intermediate activations (no grad tracking inside); saves only inputs. Backward re-runs forward to regenerate activations, then backprops.
- checkpoint_sequential(functions, segments, input): segment_size = len(functions)//segments; checkpoint each of the first (segments-1) segments via run_function(start,end) closure; the LAST segment runs without checkpointing.
  - run_function(start,end,functions): closure applying modules start..end sequentially.
- Boundary forward (between segments) runs under no_grad in the reentrant path; recompute uses saved RNG state for correctness (preserve_rng_state) so dropout etc. match.

## Design-decision -> why
- Drop & recompute (vs swap to CPU / model parallel): no extra communication / PCIe bandwidth; orthogonal and combinable. Pure compute-for-memory trade.
- k=sqrt(n) segments: the unique minimizer of n/k + k; balances boundary-storage memory against per-segment recompute memory.
- One extra forward only: each layer recomputed at most once => total forward work doubled but backward dominates, so ~30% slowdown.
- Keep expensive ops, drop cheap ones: minimizes recompute cost for given memory saving.
- Static allocation + liveness sharing first, then drop: the drop plan composes with inplace/sharing for max saving; static plan gives exact memory cost to search B.
- Recompute must reproduce exact activations: save & restore RNG state so stochastic ops (dropout, BN noise) are identical on recompute -> exact gradients.

## Scaffold correspondence
- Pre-method scaffold: a deep sequential net + standard train loop where forward stores all activations for backward (autograd default). The "how backward gets its activations" is the slot.
- Final code: checkpoint_sequential splitting the layer list into k=sqrt(L) segments, checkpoint() wrapper that saves only segment inputs and recomputes the segment forward in backward.

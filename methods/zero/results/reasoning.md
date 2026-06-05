OK, let me think this through from scratch. The thing I keep running into is brutally simple: models get better as they get bigger, and I want to train one with tens of billions of parameters, eventually a trillion. But a single GPU has 32 GB. A 1.5B model doesn't even fit for *training*. So the first question isn't "what clever algorithm" — it's literally, where does all the memory go? Let me actually count.

Mixed precision, Adam, on a tensor-core GPU. Forward and backward run in fp16 because that's what the tensor cores want. So I hold the parameters in fp16: 2Ψ bytes for Ψ parameters. I hold the gradients in fp16: another 2Ψ. Fine, that's 4Ψ, and for 1.5B that's 6 GB. Annoying but survivable. The problem is what Adam needs to actually *apply* the update without the fp16 noise destroying convergence. Adam keeps a master copy of the weights in fp32 — 4Ψ — and its two running statistics, the momentum and the variance of the gradients, both fp32, so 4Ψ and 4Ψ. Let me call the optimizer-state multiplier K. Here K = 4 + 4 + 4 = 12. Total: 2Ψ (fp16 params) + 2Ψ (fp16 grads) + 12Ψ (fp32 master + momentum + variance) = 16Ψ.

Sixteen bytes per parameter. For 1.5B that's 24 GB — and the fp16 weights I actually compute with are only 3 GB of that. So twelve of those sixteen bytes are optimizer states sitting there for the whole step, used once, at the very end, when I call step(). That's the smell. The largest chunk of memory is the part that's idle 99% of the iteration.

Now, how do people scale today, and why does each one disappoint me?

Data parallelism. Replicate the whole model on every GPU, give each a different slice of the batch, run forward/backward locally, then all-reduce the gradients so everyone has the average, then everyone applies the same update to their own replica. I love DP's *efficiency*: each GPU runs the entire model over a full local batch, so the arithmetic is coarse-grained and the GPU is busy; and the only communication is one gradient all-reduce per step. Coarse compute, cheap comms — that's why DP scales. But the memory is a disaster: every single GPU holds the entire 16Ψ. Adding GPUs gives me more aggregate FLOPs but does *nothing* for per-device memory. So DP caps out around 1.4B parameters on 32 GB and that's the end of it.

Model parallelism, the tensor-slicing kind, Megatron-style. Split each layer's matmuls across GPUs. Now the model states *are* divided, so per-device memory finally drops as I add GPUs — exactly what DP can't do. But the price is steep. To compute its slice of a linear layer, each GPU needs the *whole* input activation, so the activations have to be all-reduced between sub-layers — communication inside every single transformer block, twice in the forward alone. That's fine inside one DGX-2 node where the GPUs are wired together with fat NVLink, but the moment I cross a node boundary the per-layer all-reduces and the now-fine-grained matmuls starve the hardware. I measured a 40B model split across two nodes: about 5 TFLOPs per V100, under 5% of peak. Useless.

Pipeline parallelism splits the layer *stack* into stages instead of slicing inside layers, with cheaper boundary-only communication, but it brings pipeline bubbles and needs the stages load-balanced and the model surgically restructured.

So I'm stuck between two bad options. DP: efficient, memory-blind. MP: memory-frugal, efficiency-fragile across nodes and a pain to use. Let me stare at *why* they're opposites, because the contrast feels like it's hiding the answer.

DP *replicates* model states — that's pure redundancy. N_d copies of the exact same 16Ψ. MP *partitions* model states — no redundancy, but it pays in fine-grained compute and per-layer communication. And here's the thing both of them share, the thing nobody questions: each GPU keeps all the state it's responsible for resident for the *entire* training step. DP keeps the whole model resident because it computes the whole model. MP keeps its slice resident because it computes that slice every layer.

But do I actually *need* a piece of state resident the whole time? Take layer 7's parameters. They're needed during layer 7's forward and during layer 7's backward. That's it. The rest of the step they're dead weight. The optimizer's momentum and variance for layer 7 are needed only inside step(), once, at the very end. So the assumption "store everything everywhere, all the time" is doing nothing for me except eating memory.

Here's the reframing. DP's redundancy is the problem; MP's fine-grained compute is the problem. What if I keep DP's coarse-grained computation — every GPU still runs the whole model over its own batch slice, so the FLOPs stay big and the GPU stays busy — but I *refuse to replicate the model states*? Instead of every GPU holding all 16Ψ, I *partition* the 16Ψ across the N_d data-parallel GPUs, the way MP partitions, but without changing how the computation is sliced. Each GPU is the *owner* of 1/N_d of the model states. When a GPU transiently needs state it doesn't own, it gets it from the owner — and throws it away the moment it's done.

That decouples two things that DP and MP both glued together: *where state is stored* (partitioned, one owner each) versus *where state is computed* (everywhere, coarse-grained, like DP). If I can pull that off without blowing up communication, I get MP's memory scaling and DP's efficiency at the same time. Let me see if the communication actually works out, because that's where this could die.

Let me attack the 16Ψ in pieces, easiest first. The optimizer states are the fattest — 12 of the 16 bytes — and they're touched only in step(). Start there.

Partition the optimizer states into N_d equal chunks. GPU i owns chunk i: the fp32 master weights, momentum, and variance for 1/N_d of the parameters, and *only* that chunk. So GPU i can only run the Adam update for *its* 1/N_d of the parameters. Memory for model states drops from 4Ψ + KΨ to 4Ψ + KΨ/N_d. (The 4Ψ is the fp16 params plus fp16 grads, still replicated for now; the KΨ optimizer states are now sharded.) For large N_d that's about 4Ψ — a 4× reduction. For a 7.5B model on 64-way DP that's about 31 GB per device, versus 120 GB replicated. Already it fits.

But wait — if GPU i only updates 1/N_d of the parameters, then after step() each GPU has fresh values for only its slice. For the next forward pass everyone needs *all* the parameters again. So after the update I do an all-gather of the fp16 parameters: each GPU broadcasts its freshly-updated slice, everyone collects the full set. Cost: Ψ. And how did everyone get the averaged gradients to do their slice's update? The gradient all-reduce, as in normal DP, costs 2Ψ. Hmm, but actually I should be careful here and recount communication once I also touch the gradients, because the all-reduce might be doing more work than I need.

Let me reconsider the gradients. With optimizer states partitioned, GPU i only updates parameters in partition i, so it only needs the *reduced* gradients for partition i. Why am I all-reducing the full gradient so that *every* GPU ends up with the full averaged gradient? That's exactly the replication I'm trying to kill. As each layer's gradients become ready in the backward pass, I should reduce them *only onto the GPU that owns that partition*, and then immediately free that gradient's memory everywhere else. That's not an all-reduce; that's a reduce-scatter — different partitions reduced onto different owners. Now the gradient memory per device drops from 2Ψ to 2Ψ/N_d.

So combining optimizer-state and gradient partitioning, per-device model-state memory is 2Ψ (still-replicated fp16 params for the forward) + 14Ψ/N_d (sharded fp16 grads + fp32 master + momentum + variance), which for large N_d is about 2Ψ — an 8× reduction. For that 7.5B model on 64-way DP, about 16.6 GB per device.

Now the crucial worry: did I just blow up communication? Let me recount carefully, because this is where the whole idea lives or dies. Baseline DP moves 2Ψ per step — the all-reduce, which is really a reduce-scatter (Ψ) followed by an all-gather (Ψ). My version: I reduce-scatter the gradients onto their owners — Ψ. Each owner updates its parameter slice. Then I all-gather the updated fp16 parameters so everyone has the full model for the next forward — Ψ. Total Ψ + Ψ = 2Ψ. *Exactly the same as baseline DP.* I just split the all-reduce into its two natural halves — reduce-scatter the grads, all-gather the params — and stuck the optimizer step in the middle. The momentum and variance never move; they stay pinned to their owner, used locally in step(). I got an 8× memory cut for *zero* extra communication. That's the moment this stops being a trick and starts being obviously right.

Now the last and scariest piece: the parameters themselves. So far the fp16 parameters are still replicated — that stubborn 2Ψ that won't shrink. To make memory scale *linearly* with N_d, all the way down, I have to partition those too. So GPU i stores only the fp16 parameters of partition i. Then per-device model-state memory becomes 16Ψ/N_d — linear in the number of devices. For 64-way DP on the 7.5B model, about 1.9 GB. And the punchline: there's no longer any fixed per-device floor. As long as I have enough GPUs to share the 16Ψ, I can fit a model of *any* size. A trillion parameters needs ~16 TB of model states; on 1024 GPUs that's 16 GB each — fits in a 32 GB GPU with room for the rest.

But now the obvious objection screams at me: if each GPU only stores its own slice of the parameters, how does it run the *whole* model in the forward pass? It needs layer 7's params to compute layer 7, and it might not own layer 7. Naively I'd have to gather all the parameters onto every GPU before the forward — which is exactly the 16Ψ replication I just killed, back through the front door. Dead end.

The way out is the temporal observation from before: layer 7's parameters are only needed *during* layer 7's computation. So I don't gather all parameters up front. I pipeline it. Walk the forward pass layer by layer; right before the computation reaches the partition that owns some layers, the *owner* of those parameters broadcasts them to everyone; every GPU uses them to compute that chunk; the moment that chunk is done, the non-owners *discard* them. So at any instant a GPU holds its own permanent slice plus one transient layer's worth of borrowed parameters — not the whole model. I'm spreading the all-gather across the entire forward pass and reclaiming the memory as I go.

What does that cost in communication? Over the whole forward, each partition's parameters get broadcast out once. Summed over all partitions that's Ψ worth of parameter movement (N_d partitions, each Ψ/N_d, broadcast — total Ψ). The backward pass needs the parameters again, in reverse order, to compute gradients — another Ψ. And I still need the reduce-scatter of the gradients onto owners — Ψ. So total communication is Ψ (forward params) + Ψ (backward params) + Ψ (gradient reduce-scatter) = 3Ψ. Baseline DP was 2Ψ. So 3Ψ/2Ψ = 1.5×. A 50% bump in communication volume, in exchange for memory that shrinks *linearly* with the number of devices, with no floor. That's a trade I take every time. And both the gradient and parameter partitioning rest on the same insight I keep coming back to: not all of the gradient and not all of the parameters are needed all the time, so I can communicate them judiciously instead of storing them redundantly.

Let me put a stake in the latency worry, because reduce-scatter-as-sequential-reduces and all-gather-as-sequential-broadcasts do add round trips. But for the models I care about — hundreds of billions of parameters — the messages are enormous, so the communication time is bound by *volume over bandwidth*, not by latency. The latency is in the noise. Good.

So the model-state side is solved: three cumulative stages of partitioning — optimizer states (4× memory, same comms), then add gradients (8× memory, same comms), then add parameters (linear in N_d, 1.5× comms). I'll call these P_os, P_g, P_p, and the whole thing the zero-redundancy data-parallel scheme — every model state has exactly one owner, no replication, computation still coarse-grained like DP.

Now the residual memory. Once model states are tamed, the next bottleneck surfaces: activations, temporary buffers, fragmentation. Let me handle each, because at 100B parameters they're not small.

Activations. Even with activation checkpointing — storing only layer-boundary activations and recomputing the rest, which buys roughly a square-root reduction at ~33% recompute overhead — a 100B model with batch 32, seq 1024, MP degree 16 still needs tens of GB per GPU for the checkpoints. And here's a redundancy I haven't touched: tensor-slicing MP *replicates* activations. When a linear layer's params are split across the MP group, every GPU in that group needs the full input activation to compute its output slice — so the checkpointed input activation is duplicated across all N_m model-parallel GPUs. That's wasteful in exactly the way the model states were. So partition the activation checkpoints across the MP group too: each MP GPU stores only 1/N_m of each checkpointed activation, and right before the backward recomputation of a layer, an all-gather rematerializes the full activation, used, then dropped. That cuts the checkpoint memory by N_m — that 100B model's ~33 GB of checkpoints drops to ~2 GB per GPU.

Does the extra all-gather hurt? In Megatron with checkpointing, each transformer block already does two all-reduces in the forward, two in the recomputation, two in the backward — and an all-reduce moves 2× the message size, so that's 12 × seq × hidden per block. My partitioned-activation scheme adds one all-gather per block before recomputation — an all-gather moves 1× the message size, so seq × hidden. That's under 10% on top of the existing MP communication. Cheap. And there's a bonus: because the activation memory now drops by N_m, I can afford a much bigger batch, and DP communication volume is inversely proportional to batch size — so a 16× bigger batch can cut DP communication by an order of magnitude. The arithmetic intensity of these big models (compute per iteration over checkpoint bytes) is ≥10K and grows with hidden dimension, so I can even offload those partitioned checkpoints to CPU and *still* hide the transfer behind compute, driving activation memory to nearly zero.

Temporary buffers. High-performance libraries fuse all the gradients into one flat buffer before an all-reduce or norm computation, because big messages get better bandwidth. But a fused fp32 buffer sized to the whole model grows with Ψ — for a 3B model that's 12 GB. The fix is mundane but necessary: cap the fused buffer at a constant size once the model gets large. The buffer no longer grows with the model, and as long as the constant is big enough, the messages are still large enough to be bandwidth-efficient.

Fragmentation. During checkpointing the forward interleaves long-lived checkpoints with short-lived discarded activations; the backward interleaves long-lived parameter gradients with short-lived activation gradients and scratch buffers. That interleaving of long- and short-lived allocations shatters memory into pieces, so an allocation can fail for lack of *contiguous* memory even when plenty is free — I've seen >30% stranded — and the allocator wastes time hunting for contiguous space. So pre-allocate contiguous chunks for the checkpoints and the gradients, and copy each into its pre-reserved chunk as it's produced. Long-lived tensors land in contiguous memory; the short-lived churn happens elsewhere. Defragmentation on the fly.

Let me write the core down. The model-state partitioning is the heart; I'll lay out the data-parallel step that owns 1/N_d of every state, reduce-scatters gradients onto owners, runs the local Adam update, and all-gathers the updated parameters — with the optional pipelined parameter gather for the P_p stage. Comments tie each block back to the reasoning.

```python
import torch
import torch.distributed as dist

class ZeroRedundancyTrainer:
    """Data-parallel training where every model state has exactly ONE owner.
    Stage controls how aggressively we partition:
      1 = P_os  (optimizer states),
      2 = P_g   (+ gradients),
      3 = P_p   (+ parameters).
    """
    def __init__(self, params, lr, stage=2):
        self.world = dist.get_world_size()
        self.rank  = dist.get_rank()
        self.stage = stage
        self.params = params                       # flat list of fp16 parameters
        self.N = sum(p.numel() for p in params)

        # Partition the parameter index space into N_d contiguous shards;
        # this rank OWNS shard `rank`. Optimizer states exist ONLY for the owned shard.
        self.shards = self._partition(self.params, self.world)
        owned = self.shards[self.rank]
        # fp32 master + momentum + variance, but only for the 1/N_d we own (P_os).
        self.fp32_master = owned.detach().float()
        self.momentum    = torch.zeros_like(self.fp32_master)
        self.variance    = torch.zeros_like(self.fp32_master)
        self.lr = lr
        self.t = 0

    def _partition(self, params, n):
        # split the flattened parameters into n equal contiguous shards
        ...

    def backward_grad_hook(self, grad_bucket, owner_rank):
        # P_g: as each bucket of gradients becomes ready, REDUCE it onto its owner
        # only (reduce-scatter), then free it everywhere else -> grad memory 2*Psi/N_d.
        dist.reduce(grad_bucket, dst=owner_rank, op=dist.ReduceOp.SUM)
        grad_bucket.div_(self.world)               # average
        if self.rank != owner_rank:
            grad_bucket.zero_()                    # release: not needed here anymore

    def gather_params_for_layer(self, layer_params, owner_rank):
        # P_p: right before a layer is computed, its OWNER broadcasts the params;
        # everyone uses them, non-owners discard right after (pipelined all-gather).
        dist.broadcast(layer_params, src=owner_rank)
        return layer_params                        # caller frees after use

    def step(self, owned_grad):
        # Adam update on the OWNED shard only -- momentum/variance never leave home.
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.momentum.mul_(b1).add_(owned_grad, alpha=1 - b1)
        self.variance.mul_(b2).addcmul_(owned_grad, owned_grad, value=1 - b2)
        m_hat = self.momentum / (1 - b1 ** self.t)
        v_hat = self.variance / (1 - b2 ** self.t)
        self.fp32_master.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-self.lr)
        # cast the freshly-updated owned shard back to fp16
        owned_fp16 = self.fp32_master.half()
        # everyone needs the FULL fp16 model for the next forward:
        # all-gather the updated shards (P_os/P_g) -- total Psi moved.
        full_fp16 = self._all_gather(owned_fp16)
        return full_fp16
```

Tie it together. The whole step in zero-redundancy DP: forward (in P_p, pulling each partition's parameters via a pipelined broadcast and discarding after use), backward (gradients reduce-scattered onto owners the instant they're ready, then freed), each owner runs Adam locally on its shard with its pinned momentum/variance, then an all-gather restores the full fp16 parameters for the next iteration. Communication is 2Ψ through P_g (identical to plain DP) and 3Ψ — 1.5× — once parameters are partitioned too; memory falls 4×, 8×, and then linearly with N_d across the three stages. On top of that, partitioned activation checkpoints (rematerialized by all-gather, optionally offloaded to CPU), constant-size fused buffers, and on-the-fly defragmentation into pre-allocated contiguous chunks clean up the residual memory. The single idea underneath all of it: store each piece of state exactly once and move it to where the computation needs it, exactly when the computation needs it — because nothing is needed everywhere, all the time.

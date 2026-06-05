# ZeRO (Zero Redundancy Optimizer)

## Problem

Training a model whose total state exceeds one device's memory, across a cluster, without sacrificing compute or communication efficiency. Mixed-precision Adam training needs 16Ψ bytes of *model states* per Ψ parameters (2Ψ fp16 params + 2Ψ fp16 grads + 4Ψ fp32 master + 4Ψ momentum + 4Ψ variance; optimizer-state multiplier K = 12). Data parallelism replicates all 16Ψ on every device (efficient but memory-blind, caps near 1.4B params on 32 GB GPUs); model parallelism partitions but suffers fine-grained compute and per-layer communication that collapses across node boundaries.

## Key idea

Data and model parallelism both keep all state resident on every device for the whole step, even though a layer's parameters are only needed during that layer's forward/backward and the optimizer moments only inside `step()`. ZeRO **partitions** the model states across the N_d data-parallel devices (one owner each, zero replication) while keeping DP's coarse-grained computation, moving each piece of state to where it is computed exactly when needed. This gives MP's memory scaling with DP's efficiency.

## Final method

**ZeRO-DP**, three cumulative stages:

- **P_os (optimizer states):** each device owns and updates 1/N_d of the fp32 master weights, momentum, variance. Model-state memory: 4Ψ + KΨ/N_d ≈ 4Ψ for large N_d → **4× reduction**. Communication unchanged.
- **P_g (+ gradients):** as each gradient becomes ready in the backward, reduce-scatter it onto its owner only, then free it elsewhere. Memory: 2Ψ + 14Ψ/N_d ≈ 2Ψ → **8× reduction**. Communication still 2Ψ (reduce-scatter Ψ + all-gather of updated params Ψ = identical to baseline DP's all-reduce, which is itself reduce-scatter + all-gather).
- **P_p (+ parameters):** each device stores only its 1/N_d of the fp16 parameters; during forward, the owner of each partition broadcasts its params (pipelined across the pass), non-owners discard after use; repeated in reverse for backward. Memory: 16Ψ/N_d → **linear in N_d, no floor**. Communication: Ψ (forward params) + Ψ (backward params) + Ψ (gradient reduce-scatter) = 3Ψ = **1.5× baseline DP**.

With all three on, a trillion-parameter model (~16 TB of states) fits on 1024 GPUs (16 GB each).

**ZeRO-R** (residual states): (1) **P_a** — partition activation checkpoints across the MP group (each MP rank stores 1/N_m), rematerialize with all-gather before backward recompute (adds <10% to MP communication); optionally offload partitioned checkpoints to CPU (large arithmetic intensity hides the transfer). (2) **C_B** — cap fused communication buffers at a constant size so they don't grow with Ψ. (3) **M_D** — defragment on the fly by copying long-lived checkpoints/gradients into pre-allocated contiguous chunks.

## Code

```python
import torch
import torch.distributed as dist

class ZeroRedundancyTrainer:
    """DP where every model state has exactly one owner.
    stage: 1=P_os, 2=P_g (+grads), 3=P_p (+params)."""
    def __init__(self, params, lr, stage=2):
        self.world = dist.get_world_size()
        self.rank  = dist.get_rank()
        self.stage = stage
        self.params = params                     # fp16 parameters
        self.shards = self._partition(params, self.world)
        owned = self.shards[self.rank]
        # optimizer state ONLY for the owned 1/N_d shard (P_os)
        self.fp32_master = owned.detach().float()
        self.momentum    = torch.zeros_like(self.fp32_master)
        self.variance    = torch.zeros_like(self.fp32_master)
        self.lr = lr
        self.t = 0

    def _partition(self, params, n):
        ...  # flatten and split into n equal contiguous shards

    def backward_grad_hook(self, grad_bucket, owner_rank):
        # P_g: reduce-scatter each gradient bucket onto its owner, free elsewhere
        dist.reduce(grad_bucket, dst=owner_rank, op=dist.ReduceOp.SUM)
        grad_bucket.div_(self.world)
        if self.rank != owner_rank:
            grad_bucket.zero_()

    def gather_params_for_layer(self, layer_params, owner_rank):
        # P_p: owner broadcasts a partition's params just before it is computed;
        # non-owners discard after use (pipelined all-gather over the forward)
        dist.broadcast(layer_params, src=owner_rank)
        return layer_params

    def step(self, owned_grad):
        # local Adam on the owned shard; momentum/variance never move
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.momentum.mul_(b1).add_(owned_grad, alpha=1 - b1)
        self.variance.mul_(b2).addcmul_(owned_grad, owned_grad, value=1 - b2)
        m_hat = self.momentum / (1 - b1 ** self.t)
        v_hat = self.variance / (1 - b2 ** self.t)
        self.fp32_master.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-self.lr)
        owned_fp16 = self.fp32_master.half()
        return self._all_gather(owned_fp16)      # restore full fp16 model

    def _all_gather(self, owned_fp16):
        out = [torch.empty_like(owned_fp16) for _ in range(self.world)]
        dist.all_gather(out, owned_fp16)
        return torch.cat(out)
```

In practice this is the DeepSpeed `zero_optimization` config: stages 1/2/3 select P_os / P_g / P_p, with gradient bucketization, partitioned activation checkpointing, constant-size buffers, and contiguous-memory defragmentation handled by the runtime.

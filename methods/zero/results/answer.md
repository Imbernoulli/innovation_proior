# ZeRO (Zero Redundancy Optimizer)

## Problem

Training a model whose total state exceeds one device's memory, across a cluster, without sacrificing compute or communication efficiency. Mixed-precision Adam training needs 16Ψ bytes of *model states* per Ψ parameters (2Ψ fp16 params + 2Ψ fp16 grads + 4Ψ fp32 master + 4Ψ momentum + 4Ψ variance; optimizer-state multiplier K = 12). Data parallelism replicates all 16Ψ on every device (efficient but memory-blind, caps near 1.4B params on 32 GB GPUs); model parallelism partitions but suffers fine-grained compute and per-layer communication that collapses across node boundaries.

## Key idea

Data and model parallelism both keep all state resident on every device for the whole step, even though a layer's parameters are only needed during that layer's forward/backward and the optimizer moments only inside `step()`. ZeRO **partitions** the model states across the N_d data-parallel devices (one owner each, zero replication) while keeping DP's coarse-grained computation, moving each piece of state to where it is computed exactly when needed. This gives MP's memory scaling with DP's efficiency.

## Final method

**ZeRO-DP**, three cumulative stages:

- **P_os (optimizer states):** each device owns and updates 1/N_d of the fp32 master weights, momentum, and variance. Model-state memory is (2 + 2 + K/N_d)Ψ = (4 + 12/N_d)Ψ for mixed-precision Adam, approaching 4Ψ for large N_d: a **4× reduction**. Communication stays at baseline volume by using the reduce-scatter half of the gradient all-reduce, applying the local update, then using the all-gather phase for updated fp16 parameters.
- **P_os+g (+ gradients):** each gradient bucket is reduced to the owner of the corresponding parameter shard and released elsewhere as soon as it is no longer needed. Memory is (2 + (2 + K)/N_d)Ψ = (2 + 14/N_d)Ψ, approaching 2Ψ: an **8× reduction**. Communication is still 2Ψ (gradient reduce-scatter Ψ + updated-parameter all-gather Ψ), identical to baseline DP's reduce-scatter + all-gather all-reduce volume.
- **P_os+g+p (+ parameters):** each device stores only its 1/N_d of the fp16 parameters. During forward and backward, owners broadcast the needed parameter shards in model order, and non-owners discard each shard after use. Memory is (2 + 2 + K)Ψ/N_d = 16Ψ/N_d: **linear in N_d, no fixed floor**. Communication is Ψ for forward parameter materialization + Ψ for backward parameter materialization + Ψ for gradient reduce-scatter = 3Ψ = **1.5× baseline DP**.

With all three on, a trillion-parameter model (~16 TB of states) fits on 1024 GPUs (16 GB each).

**ZeRO-R** (residual states): (1) **P_a** — partition activation checkpoints across the MP group (each MP rank stores 1/N_m), rematerialize with all-gather before backward recompute (adds <10% to MP communication); optionally offload partitioned checkpoints to CPU (large arithmetic intensity hides the transfer). (2) **C_B** — cap fused communication buffers at a constant size so they don't grow with Ψ. (3) **M_D** — defragment on the fly by copying long-lived checkpoints/gradients into pre-allocated contiguous chunks.

## Code

```python
import torch
import torch.distributed as dist

class ZeroRedundancyTrainer:
    """DP where every model state has exactly one owner.
    stage: 1=P_os, 2=P_os+g, 3=P_os+g+p."""
    def __init__(self, flat_fp16_params, lr, stage=2):
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.stage = stage
        self.numel = flat_fp16_params.numel()
        self.chunk = (self.numel + self.world - 1) // self.world

        if stage < 3:
            self.full_fp16_padded = self._pad(flat_fp16_params.detach().clone())
            self.owned_fp16 = self.full_fp16_padded.narrow(0, self.rank * self.chunk,
                                                           self.chunk)
            self.full_fp16 = self._trim(self.full_fp16_padded)
        else:
            self.full_fp16_padded = None
            self.owned_fp16 = self._chunks(flat_fp16_params)[self.rank].contiguous()
            self.full_fp16 = None

        self.fp32_master = self.owned_fp16.float()
        self.momentum = torch.zeros_like(self.fp32_master)
        self.variance = torch.zeros_like(self.fp32_master)
        self.lr = lr
        self.t = 0

    def _pad(self, flat_tensor):
        flat = flat_tensor.contiguous().view(-1)
        total = self.chunk * self.world
        if flat.numel() == total:
            return flat
        return torch.cat([flat, flat.new_zeros(total - flat.numel())])

    def _chunks(self, flat_tensor):
        return list(self._pad(flat_tensor).split(self.chunk))

    def _trim(self, flat_tensor):
        return flat_tensor[:self.numel].contiguous()

    def reduce_scatter_owned_grad(self, flat_local_grad):
        # Each rank contributes all local gradient shards; each owner receives one sum.
        chunks = self._chunks(flat_local_grad.float())
        owned_grad = torch.empty_like(self.fp32_master)
        dist.reduce_scatter(owned_grad, chunks, op=dist.ReduceOp.SUM)
        return owned_grad.div_(self.world)

    def step(self, flat_local_grad):
        owned_grad = self.reduce_scatter_owned_grad(flat_local_grad)
        # local Adam on the owned shard; momentum/variance never move
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.momentum.mul_(b1).add_(owned_grad, alpha=1 - b1)
        self.variance.mul_(b2).addcmul_(owned_grad, owned_grad, value=1 - b2)
        m_hat = self.momentum / (1 - b1 ** self.t)
        v_hat = self.variance / (1 - b2 ** self.t)
        self.fp32_master.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-self.lr)
        self.owned_fp16.copy_(self.fp32_master.to(torch.float16))
        if self.stage < 3:
            return self._all_gather_params()
        return self.owned_fp16

    def _all_gather_params(self):
        # P_os/P_os+g: updated fp16 shards replace the all-gathered gradient replica.
        out = [torch.empty_like(self.owned_fp16) for _ in range(self.world)]
        dist.all_gather(out, self.owned_fp16)
        self.full_fp16_padded = torch.cat(out)
        self.owned_fp16 = self.full_fp16_padded.narrow(0, self.rank * self.chunk,
                                                       self.chunk)
        self.full_fp16 = self._trim(self.full_fp16_padded)
        return self.full_fp16

    def materialize_parameter_shard(self, owner_rank):
        # P_os+g+p: sequential broadcasts form the pipelined parameter all-gather.
        if owner_rank == self.rank:
            shard = self.owned_fp16.detach().clone()
        else:
            shard = torch.empty(self.chunk, device=self.owned_fp16.device,
                                dtype=self.owned_fp16.dtype)
        dist.broadcast(shard, src=owner_rank)
        start = owner_rank * self.chunk
        end = min(start + self.chunk, self.numel)
        return shard[:max(0, end - start)]


class PartitionedActivationCheckpoints:
    def __init__(self, mp_group, partition_dim=-1):
        self.group = mp_group
        self.rank = dist.get_rank(group=mp_group)
        self.world = dist.get_world_size(group=mp_group)
        self.partition_dim = partition_dim
        self.store = {}

    def save(self, key, activation):
        chunks = activation.detach().chunk(self.world, dim=self.partition_dim)
        self.store[key] = chunks[self.rank].contiguous()

    def gather_for_recompute(self, key):
        local = self.store[key]
        parts = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(parts, local, group=self.group)
        return torch.cat(parts, dim=self.partition_dim)


def constant_fused_buffer(required_elements, cap_elements, dtype, device):
    return torch.empty(min(required_elements, cap_elements), dtype=dtype, device=device)


class ContiguousTensorStore:
    def __init__(self, num_elements, dtype, device):
        self.storage = torch.empty(num_elements, dtype=dtype, device=device)
        self.offset = 0

    def append(self, tensor):
        view = self.storage[self.offset:self.offset + tensor.numel()].view_as(tensor)
        view.copy_(tensor)
        self.offset += tensor.numel()
        return view
```

The implementation target is a PyTorch-compatible training wrapper: bucketize gradients by owner, reduce-scatter them, update only the owned optimizer shard, all-gather updated fp16 parameters for P_os/P_os+g, and switch to broadcast-on-demand parameter materialization for P_os+g+p. Residual memory is handled by partitioned activation checkpoints, constant-size communication buffers, and contiguous storage for long-lived checkpoints and gradients.

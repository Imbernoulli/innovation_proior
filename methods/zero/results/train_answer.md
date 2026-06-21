Transformer language models reliably get better as they get bigger, so we want to train models with tens of billions of parameters and eventually a trillion, but a single accelerator holds only 16–32 GB and even a 1.5B-parameter model does not fit for *training*. The reason is not the weights themselves but the entourage of per-parameter state. Counting carefully under mixed-precision Adam — the standard recipe on tensor-core GPUs — the forward and backward run in fp16, so we hold an fp16 copy of the parameters ($2\Psi$ bytes) and an fp16 copy of the gradients ($2\Psi$), but to apply updates without fp16 noise wrecking convergence Adam keeps fp32 master state: an fp32 master copy of the weights ($4\Psi$), the fp32 first moment / momentum ($4\Psi$), and the fp32 second moment / variance ($4\Psi$). With the optimizer-state multiplier $K = 4 + 4 + 4 = 12$, the total model-state memory is $2\Psi + 2\Psi + K\Psi = 16\Psi$ bytes — twenty-four GB for a 1.5B model, of which the fp16 weights we actually compute with are only $\sim 3$ GB. Twelve of those sixteen bytes are optimizer states that sit idle the entire iteration and are touched exactly once, in `step()`. That is the smell: the largest chunk of memory is the part doing nothing 99% of the time.

The two ways people scale today each disappoint for a precise reason. Data parallelism replicates the whole model on every device, gives each a different batch slice, runs forward/backward locally, and all-reduces the gradients so everyone applies the same update. Its computation is coarse-grained (each device runs the whole model over a full local batch, so the GPU stays busy) and its communication is cheap (one gradient all-reduce per step, volume $2\Psi$ as a reduce-scatter of $\Psi$ followed by an all-gather of $\Psi$) — that is why it scales efficiently. But every device holds the entire $16\Psi$; adding devices buys aggregate FLOPs and *nothing* for per-device memory, so DP caps out near 1.4B parameters on 32 GB and stops. Model parallelism of the tensor-slicing kind (Megatron) splits each layer's matmuls across devices, so the model states finally *do* partition and per-device memory drops — but to compute its slice of a linear layer each device needs the whole input activation, forcing an all-reduce of activations inside every transformer block, twice in the forward alone. That is tolerable inside one node on fat NVLink and catastrophic across node boundaries: a 40B model split across two DGX-2 nodes measured about 5 TFLOPs per V100, under 5% of peak. Pipeline parallelism splits the layer stack into stages with cheaper boundary-only communication but brings pipeline bubbles, load-balancing, and model surgery. So we are stuck between DP (efficient, memory-blind) and MP (memory-frugal, efficiency-fragile across nodes).

Staring at *why* they are opposites reveals the answer. DP replicates model states — $N_d$ identical copies of the same $16\Psi$, pure redundancy. MP partitions with no redundancy but pays in fine-grained compute and per-layer communication. Both share one unexamined assumption: each device keeps all the state it is responsible for resident for the *entire* step. But layer 7's parameters are needed only during layer 7's forward and backward, and the momentum and variance only inside `step()`, once. Nothing is actually needed everywhere, all the time. I propose **ZeRO**, the Zero Redundancy Optimizer: keep DP's coarse-grained computation — every device still runs the whole model over its own batch slice — but refuse to replicate the model states. Partition the $16\Psi$ across the $N_d$ data-parallel devices the way MP partitions state, *without* changing how the computation is sliced, so each device is the sole owner of $1/N_d$ of the model states and fetches anything it transiently needs from the owner, discarding it the instant it is done. This decouples *where state is stored* (partitioned, one owner each) from *where state is computed* (everywhere, coarse-grained, like DP), and if the communication survives, it delivers MP's memory scaling with DP's efficiency.

The model-state side is **ZeRO-DP**, built in three cumulative stages, attacking the $16\Psi$ fattest-first. The optimizer states are 12 of the 16 bytes and touched only in `step()`, so partition them: device $i$ owns chunk $i$ of the fp32 master weights, momentum, and variance for $1/N_d$ of the parameters and only that chunk, so it can only run the Adam update for its $1/N_d$. This is **P_os**, and model-state memory drops from $(2 + 2 + K)\Psi$ to $(2 + 2 + K/N_d)\Psi = (4 + 12/N_d)\Psi$, approaching $4\Psi$ for large $N_d$ — a 4× reduction. After `step()` each device has fresh values for only its slice, so the next forward needs all parameters again; the schedule that makes this free is to notice a bandwidth-optimal all-reduce *already is* a reduce-scatter followed by an all-gather. We do not need the all-gathered gradient — each owner needs only its reduced gradient shard — so we stop after the reduce-scatter ($\Psi$), run the owner's local optimizer step, and spend the second $\Psi$ movement all-gathering the *updated fp16 parameters* instead of gathering gradients. Two $\Psi$-sized phases, with the Adam step inserted between them: communication stays at the baseline $2\Psi$.

Next the gradient memory. Since device $i$ only updates partition $i$, it only ever needs the *reduced* gradients for partition $i$; a full $2\Psi$ gradient tensor resident to the end is just another replica. So as each layer's gradient bucket becomes ready in the backward pass, reduce it onto the owner of that parameter partition and immediately free it elsewhere — the same reduce-scatter, scheduled at bucket boundaries instead of after the whole backward. This is **P_os+g**, dropping gradient memory from $2\Psi$ to $2\Psi/N_d$ and giving model-state memory $(2 + (2 + K)/N_d)\Psi = (2 + 14/N_d)\Psi$, approaching $2\Psi$ — an 8× reduction — for *zero* extra communication, still $2\Psi$ (gradient reduce-scatter $\Psi$ + updated-parameter all-gather $\Psi$). The momentum and variance never move; they stay pinned to their owner and are used locally. An 8× memory cut for free is the moment this stops being a trick and becomes obviously right.

The last and scariest piece is the fp16 parameters themselves — the stubborn $2\Psi$ that will not shrink. Partitioning them so device $i$ stores only partition $i$ gives memory $(2 + 2 + K)\Psi/N_d = 16\Psi/N_d$, *linear in $N_d$ with no fixed floor*: a trillion-parameter model's $\sim 16$ TB of states fits on 1024 GPUs at 16 GB each. The obvious objection is that a device storing only its slice cannot run the whole forward — it needs layer 7's parameters even if it does not own them, and gathering all parameters up front is exactly the $16\Psi$ replication we killed. The escape is again temporal: layer 7's parameters are needed only *during* layer 7's computation, so we pipeline the gather. Walking the forward layer by layer, right before the computation reaches the layers a partition owns, that owner *broadcasts* its parameters to everyone, every device computes that chunk, and non-owners *discard* the borrowed parameters the moment the chunk is done. At any instant a device holds its permanent slice plus one transient layer's worth — never the whole model. This is **P_os+g+p**. The cost recount is decisive: over the whole forward each partition's parameters are broadcast once, summing to $\Psi$; the backward needs them again in reverse order for another $\Psi$; the gradient reduce-scatter onto owners is $\Psi$; total $\Psi + \Psi + \Psi = 3\Psi$ against baseline DP's $2\Psi$, a 1.5× bump in exchange for memory that shrinks linearly with no floor — a trade taken every time. The reduce-scatter-as-sequential-reduces and all-gather-as-sequential-broadcasts do add round trips, but for hundred-billion-parameter models the messages are enormous, so communication is bound by volume over bandwidth and the latency is in the noise.

Taming model states surfaces the **residual states**, handled by **ZeRO-R**. Even with activation checkpointing — storing only layer-boundary activations and recomputing the rest for a roughly square-root reduction at $\sim 33\%$ recompute overhead — a 100B model still needs tens of GB of checkpoints, and tensor-slicing MP *replicates* them: every device in the MP group needs the full input activation to compute its output slice, so each checkpoint is duplicated $N_m$ times. So **P_a** partitions the activation checkpoints across the MP group — each rank stores $1/N_m$ of each checkpoint and an all-gather rematerializes the full activation right before backward recomputation, cutting checkpoint memory by $N_m$ (that 100B model's $\sim 33$ GB drops to $\sim 2$ GB per GPU). The added all-gather is under 10% on top of the all-reduces a checkpointed Megatron block already pays, and because activation memory falls by $N_m$ we can afford a much larger batch, cutting the DP communication paid per sample; the arithmetic intensity ($\geq 10$K and growing with hidden dimension) is high enough that the partitioned checkpoints can even be offloaded to CPU with the transfer hidden behind compute. **C_B** caps the fused communication buffer at a constant size once the model is large, so it stays bandwidth-efficient without growing with $\Psi$. **M_D** defragments on the fly by pre-allocating contiguous chunks for long-lived checkpoints and gradients and copying each into its reserved chunk as it is produced, so the interleaving of long- and short-lived allocations (which can strand more than 30% of memory) no longer shatters it. The single idea under all of it: store each piece of state exactly once and move it to where the computation needs it, exactly when the computation needs it — because nothing is needed everywhere, all the time.

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

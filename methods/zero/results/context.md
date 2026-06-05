# Context

## Research question

Transformer language models keep getting bigger, and bigger reliably means better: as parameter counts climb from hundreds of millions toward tens of billions and beyond, quality keeps improving. But the hardware does not cooperate. A single accelerator has on the order of 16–32 GB of memory, and a model with even a couple of billion parameters does not fit in it for *training* — not because the weights are large, but because training drags along a large entourage of per-parameter state.

The precise problem: **how do we train a model whose total state vastly exceeds one device's memory, across a cluster of devices, without (a) blowing up the per-device memory, (b) tanking compute efficiency, or (c) tanking communication efficiency?** A solution has to make the per-device memory shrink as we add devices, while keeping the arithmetic on each device coarse-grained enough to hit high throughput and keeping the bytes moved between devices small enough that the interconnect is not the bottleneck. Trading any one of those three away for the others is the easy, useless answer (e.g. push everything to CPU, or slice the model so finely that the GPUs starve). The goal is to get memory scalability *and* keep efficiency.

## Background

**Where the memory actually goes.** Consider mixed-precision training with the Adam optimizer, the standard recipe on tensor-core GPUs. Forward and backward run in fp16, so we hold an fp16 copy of the parameters (2Ψ bytes for Ψ parameters) and an fp16 copy of the gradients (2Ψ bytes). But to apply updates accurately, the optimizer keeps fp32 master copies: an fp32 copy of the parameters (4Ψ), plus Adam's two moments — the fp32 first moment / momentum (4Ψ) and the fp32 second moment / variance (4Ψ). Call the optimizer-state multiplier K; for mixed-precision Adam K = 12. Total model-state memory is 2Ψ + 2Ψ + KΨ = 16Ψ bytes. A 1.5B-parameter model therefore needs ~24 GB just for model states — far more than the ~3 GB the fp16 weights alone occupy. The optimizer states dominate.

Beyond model states there are **residual states**: activations (proportional to layers × hidden × sequence × batch — tens of GB for billion-scale models), temporary buffers (operations like gradient all-reduce fuse everything into one flat fp32 buffer, which for a few-billion-parameter model is several GB), and fragmented memory (interleaving long-lived and short-lived allocations during checkpointing/backward can strand >30% of memory).

**Activation checkpointing** (re-materialization) reduces activation memory to roughly the square root of total activations by storing only layer-boundary activations and recomputing the rest in the backward pass, at ~33% recompute overhead. Helpful but insufficient at scale — a 100B model still needs tens of GB of activation memory even with checkpointing.

**Data parallelism (DP).** The model is replicated on every device; each device processes a different slice of the mini-batch, runs forward/backward locally, and the gradients are averaged across devices with an all-reduce before each device applies the update to its own replica. DP has high *computational granularity* (each device runs the whole model over a full local batch) and low communication (one all-reduce of size proportional to the model per step), so it scales efficiently. Its fatal flaw: every device stores the *entire* 16Ψ of model states. The memory does not shrink as devices are added — DP runs out of memory above ~1.4B parameters on 32 GB GPUs.

**Model parallelism (MP).** The model itself is split across devices. *Horizontal* (tensor-slicing, e.g. Megatron-LM) splits individual layers' matmuls across devices; *vertical* (pipeline, e.g. GPipe) splits the layer stack into stages. Either way the model states *are* partitioned, so per-device memory drops. But tensor-slicing requires communication inside every layer (the activations have to be all-reduced between sliced sublayers), so it is only efficient *within a node* where inter-GPU bandwidth is high; across node boundaries the per-layer communication and the finer-grained compute crush throughput (a 40B model split across two nodes measured single-digit TFLOPs per V100, <5% of peak). Pipeline parallelism partitions vertically but needs careful load balancing and many micro-batches to keep the pipeline full.

**The key structural observation.** Both DP and MP keep *all* model states resident on every device for the *entire* training step, even though not all of those states are needed at all times — a layer's parameters are only needed during that layer's forward and backward. DP replicates (memory-wasteful, communication-cheap); MP partitions (memory-frugal, communication-expensive and compute-fine-grained). Neither separates "where state is stored" from "where state is needed."

## Baselines

- **Standard data parallelism** (replicated model states, all-reduce of gradients). Communication volume per step: an all-reduce is implemented as reduce-scatter then all-gather, each moving Ψ elements, so 2Ψ total. Limitation: per-device memory is fixed at 16Ψ regardless of device count; does not scale past ~1.4B params on 32 GB devices.

- **Tensor-slicing model parallelism (Megatron-LM).** Splits each transformer layer's linear projections across devices, requiring an all-reduce of the activations (size batch × seq × hidden) twice per transformer block in the forward, with matching communication in recomputation and backward. Partitions model states well but replicates *activations* (each sliced device needs the full input activation to compute its output slice) and incurs heavy per-layer communication; efficiency falls off a cliff across nodes.

- **Pipeline / vertical model parallelism (GPipe, PipeDream).** Splits the layer stack into stages on different devices, pipelining micro-batches. Partitions states, modest activation-boundary communication, but introduces pipeline bubbles and needs model-specific surgery and load balancing.

- **CPU offloading and other ad-hoc memory tricks.** Reduce GPU memory by parking state on the host, but pay a steep communication price moving state back and forth, and do not by themselves give clean linear scaling.

## Evaluation settings

- **Models / architectures.** GPT-2-style and BERT-style transformers across a wide size range — from ~1.5B up to tens and hundreds of billions of parameters — characterized by (layers, hidden dimension, attention heads). Standard mixed-precision (fp16/fp32) training with the Adam optimizer.
- **Hardware.** Clusters of NVIDIA V100 32 GB GPUs, organized as DGX-2 nodes (16 GPUs/node with high intra-node NVLink/NVSwitch bandwidth and lower inter-node bandwidth). Data-parallel degree N_d and model-parallel degree N_m as the two scaling axes.
- **Metrics.** Maximum trainable model size per configuration; per-device memory footprint of model states as a function of N_d; sustained throughput (TFLOPs per GPU and aggregate) and scaling efficiency; communication volume per training step relative to baseline DP.
- **Protocol.** Sequence length ~1024, batch sizes chosen to fit memory and stay near the critical batch size for convergence; activation checkpointing enabled for the large configurations.

## Code framework

The pre-existing building blocks: a mixed-precision Adam optimizer keeping fp32 master weights and two fp32 moments; a distributed-communication library exposing the standard collectives (`all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, point-to-point send/recv) over a process group; an autograd engine that fires a hook as each parameter's gradient becomes ready in the backward pass; and a generic data-parallel training step. The scaffold below is the ordinary replicated-DP step plus empty slots where a redundancy-removing scheme would intervene.

```python
import torch
import torch.distributed as dist

# --- existing primitives ---
world_size = dist.get_world_size()
rank = dist.get_rank()

class Adam16:
    """Mixed-precision Adam: fp16 compute copy + fp32 master params, momentum, variance."""
    def __init__(self, params, lr):
        self.fp16_params = params                      # 2*Psi bytes
        self.fp32_params = [p.detach().float() for p in params]   # 4*Psi
        self.momentum    = [torch.zeros_like(p) for p in self.fp32_params]  # 4*Psi
        self.variance    = [torch.zeros_like(p) for p in self.fp32_params]  # 4*Psi
    def step(self, grads):
        # standard Adam update on fp32 master state, then cast back to fp16
        ...

def forward_backward(model, micro_batch):
    """Run fp16 forward + backward; returns fp16 gradients (2*Psi)."""
    loss = model(micro_batch)
    loss.backward()
    return [p.grad for p in model.parameters()]

# --- the slot the contribution fills: a DP step that does NOT replicate all state ---
class DistributedTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        # TODO: decide what each device must actually STORE vs. what it must
        #       transiently HAVE in order to compute. Today every device stores
        #       the full 16*Psi. That is the redundancy to remove.

    def reduce_gradients(self, grads):
        # TODO: today this is a full all-reduce (2*Psi moved). Is the full,
        #       replicated averaged gradient actually needed on every device?
        dist.all_reduce_coalesced(grads)  # baseline
        return grads

    def parameter_step(self, grads):
        # TODO: today every device updates ALL parameters with the full optimizer
        #       state. Does every device need every optimizer state / every param?
        self.optimizer.step(grads)

    def train_step(self, micro_batch):
        grads = forward_backward(self.model, micro_batch)
        grads = self.reduce_gradients(grads)
        self.parameter_step(grads)
        # TODO: after the update, ensure every device has whatever parameters it
        #       needs for the next forward pass.

# residual-memory slots
def checkpoint_activations(layer, x):
    # TODO: store activations; today they are stored replicated across MP ranks.
    ...

def make_reduction_buffer(num_elements):
    # TODO: today a single fused buffer sized to the whole model (grows with Psi).
    return torch.empty(num_elements, dtype=torch.float32)
```

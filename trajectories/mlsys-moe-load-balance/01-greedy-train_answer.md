Serving a Mixture-of-Experts under expert parallelism, the per-layer latency is set by the
*most-loaded* GPU, not the average: the combine all-to-all cannot begin until every GPU has finished
its experts' work, and live token load is wildly skewed — a handful of hot experts soak up most of the
traffic while the rest sit nearly idle, with the hot set drifting as the input distribution changes.
With one expert per GPU the busy GPU grinds through a mountain of tokens while the others wait, and
most of the machine is wasted. The whole game is to lay out experts across GPUs so the busiest GPU is
as light as possible, recomputed cheaply and online every time the observed loads shift.

That core is the identical-machines makespan problem $P\,\|\,C_{\max}$: assign weighted jobs to
machines to minimize the maximum machine load. It is NP-hard — the two-machine case is exactly
PARTITION and the variable-machine case reduces from 3-PARTITION — so I will not get the exact optimum
in polynomial time, and since I re-run this every few minutes I need something cheap. Greedy gets
close. Plain list scheduling, dropping each job on the currently least-loaded machine, satisfies
$C \le (2 - 1/m)\cdot \mathrm{OPT}$: when the last-finishing job of size $p$ starts at time $t$ it went
on the least-loaded machine, so all $m$ machines already carry $\ge t$, giving total work
$W \ge m t + p$ and $C = t + p \le W/m + (1 - 1/m)p$, and any optimum is $\ge W/m$ and $\ge p$. Sorting
biggest-first before the same sweep — Longest Processing Time (LPT) — sharpens this to
$C \le (4/3 - 1/(3m))\cdot \mathrm{OPT}$. So largest-first, least-loaded is the right packing instinct,
and I keep it, but the hardware adds a constraint the textbook bound does not cover.

I propose the **three-stage hierarchical greedy bin-packing** placement — the EPLB reference algorithm
— built on one primitive I lean on twice. The constraint is that every GPU hosts the *same number* of
physical experts: the runtime allocates a fixed slot count per GPU, so this is not free-form makespan
but a *balanced* partition where every pack gets exactly $n/\text{num\_packs}$ items. I keep the greedy
rule and restrict the candidate packs at each step to those not already full: sort descending, and for
each item pick, among packs with a free slot, the one with the smallest current load, assign it there,
record its rank within the pack, bump that pack's load and count. That is `balanced_packing`. (When
there is exactly one item per pack there is nothing to balance, so item $i$ goes to pack $i$ at rank 0
without the inner loop.) The loop is irreducibly sequential — each placement's "least loaded" reads the
running loads of all earlier placements — so I move the sorted indices to the CPU and let a Python loop
carry the loads; slow, but it mirrors the greedy decisions exactly.

Greedy alone hits a wall pure packing cannot pass: it balances a *fixed* set of indivisible items, but
if one expert is hotter than the per-GPU fair share, no whole-item assignment can help. Suppose one
expert carries a fifth of all tokens across eight GPUs — perfect balance is one-eighth per GPU, but this
single set of weights cannot be split, so the GPU holding it is floored at a fifth. The fix is to make
the items divisible by *copying* them: if a hot expert has $r$ physical replicas and its requests spread
across them, each copy carries about $w_i / r$ instead of the whole $w_i$, and each extra copy lowers
the floor that expert imposes. This is the move classical scheduling never had. So a *replicate* stage
sits before packing. The runtime gives $\text{num\_phy}$ slots and $\text{num\_log}$ logical experts, so
there are exactly $\text{num\_phy} - \text{num\_log}$ extra copies, and every logical expert needs at
least one. Which expert gets each extra? Expert $i$ with count $c_i$ and load $w_i$ has per-replica load
$w_i / c_i$; one more copy drops it to $w_i / (c_i + 1)$. The per-replica load of the *current argmax* is
the ceiling, and adding a replica anywhere except an argmax leaves that ceiling untouched — so I
repeatedly feed the expert with the largest $w_i / c_i$, $\text{num\_phy} - \text{num\_log}$ times. That
is discrete water-filling, and it too is sequential (the next argmax depends on the counts updated by all
prior draws), so again Python on the CPU.

The second wall is invisible to the balance number: *where* the replicas land. If I throw all replicas
into one global pack across every GPU, greedy may scatter expert $i$'s replicas across whichever GPUs
were least loaded — across different *nodes*. The GPUs are not a flat pool; they sit in nodes with fast
intra-node NVLink and slow, scarce inter-node InfiniBand, and the router uses node-limited routing
precisely to keep the all-to-all mostly intra-node. Scattering an expert's replicas across nodes
detonates exactly the inter-node traffic the routing was built to bound — and the `locality` metric
counts this, so I cannot trade it away. The resolution is to make the placement *hierarchical*: keep a
whole expert group, and all replicas of its experts, on a single node, so locality falls out by
construction. If group membership never crosses a node boundary, an expert is never scattered.

That gives the three stages. **Stage 1** packs groups onto nodes: each group's load is the sum of its
experts' loads, so this is `balanced_packing` with items = groups, packs = nodes,
$\text{groups\_per\_node} = \text{num\_groups}/\text{num\_nodes}$ per node. **Stage 2** replicates within
each node: each node owns $\text{num\_log}/\text{num\_nodes}$ logical experts and gets
$\text{num\_phy}/\text{num\_nodes}$ slots, so the water-filling rule runs per node on its slice. **Stage
3** packs each node's replicas onto its GPUs: `balanced_packing` again, items = the node's physical
replicas weighted by per-replica load, packs = the node's GPUs, exactly
$\text{num\_replicas}/\text{num\_gpus}$ slots per GPU. Two packs with the same primitive, one
replication between them, locality for free. The global group-agnostic policy is just this hierarchy
with $\text{num\_groups} = \text{num\_nodes} = 1$ — one giant node, Stage 1 trivial — so it is one
algorithm in two regimes; here `num_nodes` divides `num_groups`, so the genuine hierarchical branch runs.

The rest is permutation bookkeeping, fiddly because each stage produces a permutation and I compose them
to emit the three maps the runtime wants: `phy2log` (logical expert per physical slot), `log2phy` (its
inverse, $-1$ padded), and `logcnt` (replica count per expert). Stage 1 relabels logical experts into a
node-major order so each node's experts form a contiguous block — a group's position is
$\text{node}\cdot\text{groups\_per\_node} + \text{rank}$, times the group size, plus the within-group
offset — which lets Stage 2 slice "this node's experts" with a reshape. Stage 3 builds each slot's final
position as $\text{gpu}\cdot\text{phy\_per\_gpu} + \text{rank}$, inverts it, then composes back: follow
each final position to its slot, to its node-major logical expert, lift node-local ids to global by
adding each node's base offset, and map node-major back to true logical with the Stage-1 inverse.
Finally I scatter each physical slot index into `log2phy` at (its logical expert, its replica rank),
leaving $-1$ where an expert has fewer than the max copies. An off-by-one here silently mis-routes
tokens, so the composition stays exact.

This is the correct, complete answer to the placement problem: it balances per-GPU load, balances
per-node load through Stage 1, and confines every expert to one node, so it should score a perfect
locality everywhere and respectable balance where the configuration is forgiving. The one thing it is
not is fast — `balanced_packing` is a Python loop over every item with an $O(P)$ min-scan inside, run
across all layers, so wall time will be on the order of hundreds of milliseconds on the larger configs,
paid on the serving critical path. That, and the balance wherever the hierarchy is starved of freedom
(the `stress-skew` config, two groups per node, tight budget, long-tail Zipf), is what the next rung
will have to read off the numbers and attack.

```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((B, n), -1, dtype=torch.int64)
    rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    for b in range(B):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        for j in range(n):
            item = sorted_idx[b, j].item()
            best = min(
                (p for p in range(num_packs) if counts[p] < items_per_pack),
                key=lambda p: loads[p],
            )
            pack_index[b, item] = best
            rank_in_pack[b, item] = counts[best]
            loads[best] += weight[b, item].item()
            counts[best] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    idx_b = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        eff = weight / logcnt.float()
        top = eff.argmax(dim=-1)
        phy2log[:, i] = top
        rank[:, i] = logcnt[idx_b, top]
        logcnt[idx_b, top] += 1
    return phy2log, rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3
    tpp = (tpm / mcnt.float()).gather(-1, p2m)
    pi, ri = balanced_packing(tpp, gpus_per_node)
    p2pp = pi * phy_per_gpu + ri
    pp2p = inv(p2pp)

    pp2m = p2m.gather(-1, pp2p)
    pp2m = (pp2m.view(L, num_nodes, -1)
            + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    pp2log = mlog2log.gather(-1, pp2m)
    pprank = prk.gather(-1, pp2p).view(L, -1)
    logcnt = mcnt.view(L, -1).gather(-1, log2mlog)

    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, pp2log * mx + pprank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return pp2log, log2phy, logcnt
```

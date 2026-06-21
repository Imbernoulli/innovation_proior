The hierarchical greedy placement is *correct* — it scored a perfect locality of 1.000 on all four
configs and solid real-model balance (0.940/0.947 on qwen3-moe, 0.925/0.931 on deepseek-v2) — but it
buried almost the entire task score in runtime: 248 ms on deepseek-v3, 102 on qwen3-moe, 153 on
deepseek-v2, 256 on stress-skew. Each config weights runtime equally with the three quality metrics and
calibrates it against the spread of the baselines, so a quarter-second placement is scored as essentially
worthless on a quarter of every config, dragging the geometric mean to 0.255. The diagnosis is not a
balance problem; the algorithm is right, it just spends hundreds of milliseconds on the serving critical
path every time the load drifts. The hierarchy, the replica-count rule, and the output maps are all worth
keeping. What I have to kill is the time, and the time lives in one routine: `balanced_packing`. It is an
outer Python loop over the $L$ layers, an inner Python loop over the $n$ items, and inside that a linear
min-scan over $P$ packs — $L\cdot n\cdot P$ interpreted iterations, each a `.item()` off a tensor — on a
machine that wants batched tensor kernels. And it is called twice, with Stage 3 the heavier (it walks
$\text{num\_replicas}$ per layer, 320 on deepseek-v3 across 61 layers); that is the 248 ms.

The obvious instinct is "vectorize the loop," and it is exactly the move that cannot be made, because the
greedy choice for item $j$ — "which pack is emptiest right now" — means *after* items $0..j{-}1$ are
placed. The decision depends on the running pack loads, which depend on every earlier decision: an
inherently sequential prefix recurrence with no closed form for the current loads short of simulating the
whole prefix. So I cannot batch away the dependency; I have to change the *rule*. And I do not actually
need the greedy's exact decisions — I need packs whose sums are about as even as the greedy's, computed
without a per-item running-load dependence. The question becomes: is there an assignment I can write as a
*fixed function of each item's sorted position alone*, no running state, that still comes out balanced? If
the pack an item goes to depends only on its rank in sorted order, I can compute it for all items at once
with index arithmetic and the sequential chain breaks.

I propose the **vectorized zigzag (snake) packing**. Start with the crudest positional rule — plain
round-robin over the sorted items: heaviest (rank 0) to pack 0, rank 1 to pack 1, …, rank $P$ back to
pack 0, so pack $p$ gets sorted-ranks $\{p, P+p, 2P+p, \dots\}$. It is trivially vectorizable and gives
exactly $n/P$ items per pack for free. But it does *not* balance: group the sorted items into consecutive
rounds of $P$; round-robin runs every round in the same direction, so pack 0 systematically collects the
heaviest element of *every* round and pack $P{-}1$ the lightest of every round. The disparities do not
cancel — they *stack*, by roughly the per-round spread times the number of rounds. That failure is
informative: the imbalance comes from every round running the same direction. So alternate it. Run even
rounds forward, packs $0,1,\dots,P{-}1$, and odd rounds *backward*, $P{-}1,\dots,1,0$. Now in round 0
pack 0 gets the heaviest item (rank 0) and pack $P{-}1$ the lightest (rank $P{-}1$); in round 1, reversed,
pack 0 gets the *lightest* of that round (rank $2P{-}1$) and pack $P{-}1$ the *heaviest* (rank $P$). Pack
0's two items are a heavy one paired with a light one; round by round, each pack alternately gets the
heavy end and the light end, so the spread within each pack *cancels* instead of accumulating. That is the
snake — reverse direction each row.

The assignment stays purely positional: the pack of rank $r$ depends only on $r$, through its round
$r // P$ and whether that round is even or odd, with within-round offset $r \bmod P$, so the pack is the
offset on even rounds and the mirror $P - 1 - \text{offset}$ on odd rounds. The within-pack rank is just
the round number, because each pack receives exactly one item per round in round order. No running loads —
so it vectorizes into one batched sort plus a handful of `arange`/`where`/`scatter` kernels whose count is
independent of $n$. The capacity constraint that the greedy enforced with a `counts[p] < items_per_pack`
filter is now automatic: with $n$ divisible by $P$ there are $n/P$ rounds, each dealing one item to each
pack, so every pack ends with exactly $n/P$ items and the hard cardinality requirement comes for free.

The real question is whether this balances as well as the greedy or trades a good heuristic for a
cheap-but-worse one. After sorting, the weights are non-increasing, so the first round is forced — greedy
places ranks $0..P{-}1$ into empty packs, the same forward sweep the snake makes. The precise claim is
conditional: when the sorted sequence keeps the packs in that alternating emptiest-bin order, the snake
*is* LPT written in closed form. A locally affine descending block has exactly this property — over two
consecutive blocks pack $p$ receives $w[p] + w[2P-1-p]$, and those pair-sums are equal across $p$, so the
backward sweep ends with the next forward sweep restarting from pack 0; smooth sorted load tails
approximate it. The condition is friendliest where the load is smooth, and Stage 3 *is* smoothed, because
replication has already split the hot experts and shaved the per-replica peaks. It is least friendly where
the sorted sequence has a cliff — a couple of huge items then a flat tail — which is exactly `stress-skew`
(Zipf at skew 0.95, only `groups_per_node = 2`). Greedy already cratered there (balance 0.222, balance_node
0.336) because the hierarchy starves Stage 1 of freedom, not because of the packing rule; the snake reads
no running loads, so it cannot do better than greedy on that cliffed sequence and may do marginally worse
on the exact node assignment. Its job is runtime, not stress-config balance, and I should not pretend it
rescues that.

Two more decisions. Both packing calls become the snake, and doing it in *both* is what matters — if I
snake-ified only Stage 1 and left Stage 3 as the loop, I would still pay the Python tax on the *larger*
packing and the 248 ms would barely move; the speedup only fully lands when every packing call is
vectorized. Stage 2, the replication, stays a sequential argmax loop: each new replica *changes* the
per-replica load of the expert it joins ($w/2$, then $w/3$, …), so the next argmax depends on the
previous draw — the diminishing-returns water-filling is the whole point. But it iterates only over the
*redundant* count $\text{num\_phy} - \text{num\_log}$, runs after the two big packings are vectorized away,
and replicating the single most-overloaded expert each time is the right rule, so I keep it. The
index-composition bookkeeping that stitches the stages — node-major relabeling, per-node gathers, slot
construction, inverses, the final scatter into `log2phy` — stays *identical*, because an off-by-one
silently mis-routes tokens; I keep the index work on the CPU too, since the sort, `where`, and scatters
are tiny relative to the model and keeping them on host avoids launching a swarm of tiny GPU kernels for
what is really index arithmetic. The only thing I replace is the body of `balanced_packing`.

So the expectations against the greedy numbers: locality stays exactly 1.000 (the hierarchy is untouched);
runtime collapses by roughly two orders of magnitude, from the 100–256 ms range into low single-digit
milliseconds; and balance lands *at or just below* greedy's on the smooth real-model configs (qwen3-moe
near 0.94, deepseek-v2 near 0.92, deepseek-v3 near its already-mediocre 0.68) while staying pinned around
0.222 on stress-skew. The single number that should move enormously is the task score: with locality held
and balance roughly held but runtime cut from a quarter-second to milliseconds, it should jump from 0.255
toward the high-0.30s — the runtime tax that crushed greedy simply removed.

```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n = weight.shape
    assert n % num_packs == 0

    if n // num_packs == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    # Sort items by weight descending
    sorted_idx = weight.float().sort(-1, descending=True).indices

    # Zigzag assignment: even blocks go 0..P-1, odd blocks go P-1..0
    positions = torch.arange(n, device=weight.device)
    block_id = positions // num_packs
    pos_in_block = positions % num_packs
    is_even = block_id % 2 == 0
    pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    rank_assign = block_id

    # Scatter back to original item order
    pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    pack_index = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    pack_index.scatter_(-1, sorted_idx, pack_expanded)
    rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)

    return pack_index.cpu(), rank_in_pack.cpu()


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

    # Stage 1: zigzag packing of groups to nodes
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: greedy replication
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: zigzag packing of replicas to GPUs
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

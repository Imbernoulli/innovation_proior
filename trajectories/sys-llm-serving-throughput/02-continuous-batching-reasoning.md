Paging fixed the memory side: the cache is scattered into blocks, waste is near zero, and I can hold a far larger set of requests' caches resident — on the A10G-7B config the sustainable batch went from ~9 max-length reservations to ~35 live-sized requests. So the question shifts. I have the *memory* to keep ~35 requests' caches live — but am I keeping ~35 requests *busy* every forward pass? If the GPU spends a step computing decode for a half-empty batch, the paged headroom buys me nothing. The new binding constraint isn't "can their caches fit," it's "how full is each forward pass."

Look at how requests flow through the engine. The standard pattern batches at the level of a whole generation call: collect a set of requests, run them together until *all* finish, then take the next set. Picture a batch where one request wants 2000 output tokens and the rest want 20. After 20 steps all but one are done — but I can't release them and admit new requests, because the batch is fixed for the duration of the call. So for the next ~1980 steps I'm running a forward pass over a batch that's almost entirely finished requests, padding, dead weight.

Quantify it, because the number tells me how big this rung is. Take the full paged batch B = 36, one request wanting 2000 output tokens and the other 35 wanting ~20 each. Time-average the *live* (unfinished) batch over the call's ~2000 steps: for the first ~20 steps all 36 are live, then for the remaining ~1980 steps exactly one is. Average live batch = (20·36 + 1980·1)/2000 ≈ 1.35 — about **3.8% occupancy**. Paging quadrupled the ceiling and call-level batching throws almost all of it away by holding it frozen: the GPU does a near-batch-of-1's worth of work while I have the memory to be doing batch-of-36. The long tail of one request starves everyone behind it in the queue.

The 2000-vs-20 split is the worst case, but the length spread the prior art calls out — prompt and generation lengths varying by orders of magnitude — guarantees the same decay in the typical case: a frozen batch of 36 is front-loaded with short requests, so it collapses to a handful of long-tail survivors early and then coasts for hundreds of steps near-empty, a time-averaged occupancy well under a third of the ceiling. The temporal waste isn't a pathology I constructed; it's what the workload's length distribution does to any frozen batch.

So the waste is now *temporal*, not spatial. Paging made the caches fit; call-level batching means a slot is occupied by a finished request that won't leave until its slowest batch-mate finishes. Two failure modes in one: finished requests can't *retire* mid-batch (they hold their slot doing nothing) and waiting requests can't *join* mid-batch (they sit in the queue though slots are effectively free) — both because the batch is frozen for the lifetime of a `generate` call. And this is an order-of-magnitude factor, not a rounding effect on top of paging's spatial win.

Could a smarter *policy* inside call-level batching rescue it? Shortest-job-first — group requests of similar target length so the tail doesn't drag a full batch — fails because I don't know the target length at arrival (it isn't known until the stop token), and even with an oracle a batch of all-2000-token requests still admits nobody until all 2000 steps finish. Smaller per-call batches waste fewer slots when stalled but directly throw away the paging headroom I just won. Both fiddle with *which* requests share a frozen batch; neither lets the batch *change* while it runs. The frozen-ness itself is the defect, so the fix has to be structural.

What's the natural unit at which the batch *could* change? A generation isn't one indivisible operation — it's a sequence of decode steps, each a forward pass producing one token per request. So reconsider the batch composition at *every step*: make scheduling decisions per **iteration**, not per generation. Before each step decide which requests run *this* step; after it, retire the ones that just produced their stop token and admit waiting ones into the freed slots. A request that finishes at step 20 is gone at step 21, its slot immediately taken by a queued request. No request ever runs a pass over a batch padded with corpses.

With retire-and-admit every step, the instant a short request finishes its slot is refilled from the queue, so the live batch stays pinned near 36 whenever there's a backlog rather than decaying to 1.35 — occupancy swings from ~3.8% back toward ~100% on exactly the same memory, purely from unfreezing the batch. That's the rung: not a new kernel, not more memory, just refusing to run passes over finished requests.

This composes with what each request needs at a given step, because requests aren't all in the same phase. A fresh request needs prefill (its whole prompt); an in-flight request needs one decode step. Iteration-level scheduling lets me drop the prefill/decode phase distinction: at each step a request just has some number of tokens still to compute — the whole prompt for a fresh request, one token for a decoding one — and the scheduler's job is to pick a set of requests and a token count for each, up to a budget, so every request's computed count creeps toward its target. Prefill and decode become the same kind of work item, "advance request r by n tokens," with different n.

That budget deserves a number, because the accounting exposes something. Call it `max_num_scheduled_tokens`, the cap on tokens committed to one forward pass. A pure-decode batch of 36 spends exactly 36 of them — one new query position per request. If the budget is 2048, then 2012 tokens of per-step capacity go untouched on every decode step. That headroom is what lets a step also carry a fresh request's whole prompt (a 1500-token prompt fits comfortably), and it's the ledger the retire/admit loop debits against. A pure-decode step leaves most of its token budget — and, as the roofline said, most of its compute — unspent; right now I only need the budget to bound and schedule the work, but that unspent capacity is a standing invitation I'll come back to.

Concretely the scheduler keeps two queues: `waiting` (arrived, not started) and `running` (cache allocated, mid-generation). Each step: walk `running`, and for each figure out how many tokens it needs — `num_tokens_with_spec − num_computed_tokens` — clamp to the remaining budget, and schedule that many, until the budget is spent. After the model runs, retire any request that emitted its stop token (free its blocks back to the pool — cheap now, just returning block numbers to the free list) and pull from `waiting` to fill the freed capacity. The batch for the next step is whatever survived plus whatever was just admitted.

One ordering decision inside the step sets the latency character of the whole server: within the budget, spend it on advancing the requests already running, or on admitting waiting ones? Admit-first maximizes how fast new work enters but can push a running request's next token behind a flood of admissions, smearing its per-token latency and blowing the fixed budget for requests already in flight. Decode-first — walk `running`, give each its ~1 token before spending any leftover budget on admissions — keeps committed requests making steady, bounded progress (their latency protected) and lets new requests fill only the genuinely spare capacity. Under a fixed-latency SLO decode-first is the right default: it bounds the tail latency of in-flight requests, which is exactly what the budget is there to protect, and admissions still happen every step there's slack. So the loop below walks `running` first, debiting the budget per decode, and admits from `waiting` only into what remains.

A memory-pressure case forces the last piece, and it's why paging being in place first matters. With requests joining and leaving every step and each growing its cache token by token, I can hit a moment where the running set needs one more block than the pool has free. Under contiguous caches this was fatal; under paging I can **preempt** — evict a request's blocks back to the pool and put it on the waiting queue to resume. Freeing is just returning block numbers, essentially free. The resume cost has two routes: recompute the victim's prefill on readmission (~2 × 512 × 7·10⁹ ≈ 7·10¹² FLOPs, ~0.1 s for a 512-token victim, needing nothing but the GPU compute I'm freeing and scaling with the victim's *prefix* length), or swap its 256 MiB of KV to host and back over PCIe (~20 ms round trip at ~25 GB/s, but it burns 256 MiB of pinned host RAM per swapped-out request and contends for the link). Both are linear in victim size and cheap relative to the many bandwidth-bound decode steps the victim still owes, which is the point. Preempt in LIFO order — the newest request — so the oldest, closest-to-finishing requests keep their progress and the youngest victim has the least to redo on resume. So the free-list and the iteration-level scheduler are made for each other: on a failed allocation the scheduler picks the newest victim, preempts it, frees its blocks, and retries, rather than crashing or stalling. Preemption stays a rare, cheap safety valve, not a hot path.

Why this raises throughput at fixed latency, beyond "no padded batches": a decode step over the batch is ~23 ms (streaming ~14 GB of weights at ~600 GB/s), essentially the same whether the live batch is 1 or 36, because decode is bandwidth-bound. So tokens/sec = live batch / 23 ms: the frozen scheme at ~1.35 live yields ~59 tokens/s, continuous batching at ~36 live yields ~1560 tokens/s on identical hardware and memory — an order-plus multiplier purely from keeping the pass full, and it stacks on paging because paging raised the ceiling. It also cuts queueing latency: under call-level batching a waiting request waits for a whole batch to drain (up to ~2000 steps in the example); under iteration-level batching it's admitted at the next step a slot opens. So serving_throughput_tokens_per_sec and requests_per_sec at the fixed budget should rise by roughly the occupancy ratio recovered (largest exactly when the workload mixes long and short generations, since that's when the frozen batch decayed most), while per-request inference_latency should *drop* as queueing collapses from batch-drain time to next-free-step time. If throughput rose but tail latency didn't, I'd suspect the admit path is starving new requests and revisit the queue policy.

The scheduler is deliberately phrased as "advance each request by some tokens toward its target, within a per-step token budget." That phrasing is the hook the next refinements plug into: the moment "how many tokens does this request need" and "how do I spend a per-step budget across requests" are first-class, I can be clever about *which* tokens to compute when (splitting a giant prefill so it doesn't monopolize a step) and *whether* I even need to compute some (reusing a shared prefix's cache). But this rung's win is removing the temporal waste: stop freezing the batch per call, re-form it every iteration, retire finished requests and admit waiting ones every step.

The core is the per-iteration scheduling loop.

```python
# vllm/v1/core/sched/scheduler.py — Scheduler.schedule() (excerpt).
# There is no "prefill phase" or "decode phase": each request just has
# num_computed_tokens vs. num_tokens_with_spec, and each step advances every
# request's computed count toward its target, within a per-step token budget.
def schedule(self) -> SchedulerOutput:
    scheduled_running_reqs: list[Request] = []
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens   # per-step token budget

    # Re-form the batch THIS step: walk the running requests.
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # How many tokens this request still needs computed:
        #   decode -> ~1 token;  unfinished prefill -> the rest of the prompt.
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        num_new_tokens = min(num_new_tokens, token_budget)  # clamp to budget

        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:
                # Out of blocks: preempt a victim (free its blocks back to the
                # pool -- cheap under paging) and retry, instead of crashing.
                preempted_req = self.running.pop()
                self._preempt_request(preempted_req, ...)
                if preempted_req == request:
                    break
            else:
                break

        scheduled_running_reqs.append(request)
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

    # (after the model runs) finished requests are retired and their blocks
    # freed; waiting requests are admitted into the freed capacity next step.
    ...
```
Paging took KV waste to near zero and bought 2–4× throughput over FasterTransformer and Orca at equal latency — the memory side is solved, and I can now hold a far larger set of requests' caches resident at once. But that shifts the binding constraint rather than removing it. Having the memory to keep, say, forty requests' caches live is worthless if I am not actually keeping forty requests *busy* on every forward pass. The standard engine batches at the granularity of a whole generation call: it collects a set of requests, runs them together until *all* of them finish, then takes the next set. Picture a batch where one request wants 2000 output tokens and the rest want 20 — after twenty steps all but one are done, but the batch is frozen for the call's duration, so for the next ~1980 steps the forward pass runs over a batch that is almost entirely finished requests, padding, dead weight. The GPU does a near-batch-of-one's worth of useful work while I have the memory for batch-of-forty. The waste is now *temporal*, not spatial: finished requests cannot retire mid-batch (they hold their slot doing nothing) and waiting requests cannot join mid-batch (they sit in the queue while slots are effectively free) — both because the batch is frozen for the lifetime of a `generate` call.

I propose **continuous (iteration-level) batching**: make scheduling decisions per **iteration** — per forward pass — rather than per generation. A generation is not one indivisible operation; it is a sequence of decode steps, each producing one token per request, so the composition of the batch can be reconsidered at every step. Before each step I decide which requests run *this* step; after the step I retire the ones that just emitted their stop token and admit waiting ones into the freed slots. The batch is re-formed every iteration, so a request that finishes at step 20 is gone at step 21 and its slot is immediately taken by a queued request — no forward pass ever runs over a batch padded with corpses. The reframing also lets me drop the prefill/decode phase distinction entirely. Instead of "the prefill batch" and "the decode batch," each request simply has a count of tokens it still needs computed — for a fresh request that is its whole prompt, for a decoding request the single next token — and the scheduler's job each step is to pick a set of requests and a token count for each, up to a per-step budget, advancing every request's computed count toward its target. Prefill and decode become the same kind of work item, "advance request $r$ by $n$ tokens this step," differing only in $n$. This is the cleaner, more general scheduler that iteration-level granularity unlocks, and — as it turns out — the hook every later refinement plugs into.

Concretely the scheduler keeps two queues, `waiting` (arrived, not yet started) and `running` (cache allocated, mid-generation). Each step it walks the `running` requests and, for each, computes how many new tokens it needs as `num_tokens_with_spec + num_output_placeholders − num_computed_tokens`, clamps that to the remaining per-step `token_budget`, and schedules that many — decoding requests need about one token, an unfinished prefill needs the rest of its prompt — continuing until the budget is spent. After the model runs, any request that emitted its stop token is retired and its blocks freed back to the pool (cheap now — it is just returning block numbers to the free list), and `waiting` requests are pulled in to fill the freed capacity. The batch for the next step is whatever survived plus whatever was admitted. The one case that needs care is memory pressure, and it is exactly why paging had to come first: with requests joining and leaving every step and each one growing its cache token by token, the running set can need one more block than the pool has free. Under contiguous caches that was fatal. Under paging I have a graceful option — **preempt** a victim: evict its blocks back to the pool and return it to the waiting queue to resume later. So when `allocate_slots` cannot satisfy a step's request, the scheduler pops a victim from `running`, preempts it, frees its blocks, and retries rather than crashing or stalling; preemption is cheap and safe only because freeing is just returning block numbers, which is what makes the iteration-level scheduler and the block pool natural partners.

What makes this raise throughput, beyond removing padded batches, is that time-averaged batch occupancy is what sets tokens/sec — decode is bandwidth-bound, so throughput tracks the *live* batch size. Per-call batching lets occupancy decay across a batch's life as members finish; iteration-level batching refills every freed slot immediately, holding occupancy at the memory ceiling continuously. That is a direct multiplier on top of paging, which is what raised the ceiling in the first place, and it isolates cleanly in measurement: against an already-continuously-batched server like TGI, the paging-plus-scheduler combination is worth 2.2–2.5× for single-completion serving and 3.3–3.5× for three parallel completions. It also cuts queueing latency, since a waiting request is admitted at the next free step rather than after a whole batch drains.

```python
# vllm/v1/core/sched/scheduler.py — Scheduler.schedule() (excerpt).
def schedule(self) -> SchedulerOutput:
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens          # per-step budget
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = (                                # decode: ~1; prefill: rest
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens)
        num_new_tokens = min(num_new_tokens, token_budget)
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:                        # out of blocks ->
                preempted_req = self.running.pop()        # preempt + free + resume later
                self._preempt_request(preempted_req, ...)
                if preempted_req == request:
                    break
            else:
                break
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1
    # finished requests retired (blocks freed), waiting requests admitted next step.
    ...
```

**Problem (from step 1).** Paging fixed the *spatial* waste (KV memory), raising the batch the GPU can hold. But the engine still batches at the granularity of a whole generation call: a batch is frozen until *all* its requests finish. So a batch with one long request and many short ones spends most of its life running a forward pass over mostly-finished requests — finished requests can't retire mid-batch (they hold their slot doing nothing) and waiting requests can't join mid-batch. The memory headroom is wasted to *temporal* under-occupancy.

**Key idea — continuous (iteration-level) batching.** Make scheduling decisions per **iteration** (per forward pass), not per generation. Before each step, choose which requests run this step; after the step, retire requests that just emitted their stop token (free their blocks) and admit waiting requests into the freed slots. The batch is re-formed every iteration, so occupancy stays near the memory ceiling continuously instead of decaying as a frozen batch drains. Drop the prefill/decode phase distinction: each request just has `num_computed_tokens` vs a target, and each step advances every request toward its target within a per-step token budget.

**Why it works.** Time-averaged batch occupancy is what sets throughput (decode is bandwidth-bound; tokens/sec tracks the *live* batch size). Per-call batching lets occupancy decay across a batch's life; iteration-level batching refills every freed slot immediately, holding occupancy at the ceiling — a direct multiplier *on top of* paging, which raised the ceiling. It also cuts queueing latency (a waiting request is admitted at the next free step, not after a whole batch drains). Memory pressure from requests joining/leaving and growing token-by-token is handled by **preemption**: when the pool can't allocate a step's blocks, evict a victim's blocks back to the pool and resume it later — cheap and safe only because freeing is just returning block numbers to the free list (paging enables it).

**Change / code.** Replace the per-call batch with a `Scheduler.schedule()` that, each step, walks the `running` queue, computes each request's needed tokens, allocates blocks within a per-step `token_budget` (preempting on exhaustion), runs the model, then retires finished requests and admits from `waiting`.

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

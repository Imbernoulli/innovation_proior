Paging fixed the memory side: the cache is scattered into blocks, waste is near zero, and I can now hold a far larger set of requests' caches resident at once. So the question shifts. I have the *memory* to keep, say, 40 requests' caches live — but am I actually keeping 40 requests *busy* every forward pass? If the GPU spends a step computing decode for a batch that's half-empty, the paged memory headroom buys me nothing. The new binding constraint isn't "can their caches fit" — it's "how full is each forward pass."

Let me look at how requests currently flow through the engine, because I think the batching granularity is wasting the headroom paging just gave me. The standard pattern batches at the level of a whole generation call: I collect a set of requests, run them together until *all of them* finish, then take the next set. Picture a batch where one request wants 2000 output tokens and the rest want 20. After 20 steps, all but one are done — but I can't release them and admit new requests, because the batch is fixed for the duration of the call. So for the next ~1980 steps I'm running a forward pass over a batch that's almost entirely finished requests, padding, dead weight. The GPU is doing a near-batch-of-1's worth of useful work while I have the memory to be doing batch-of-40. The long tail of one request starves everyone behind it in the queue.

So the waste is *temporal* now, not spatial. Paging made the caches fit; call-level batching means the batch slot is occupied by a finished request that won't leave until its batch-mates' slowest member finishes. Two failure modes in one: finished requests can't *retire* mid-batch (they hold their slot doing nothing), and waiting requests can't *join* mid-batch (they sit in the queue even though slots are effectively free). Both come from the batch being frozen for the lifetime of a `generate` call.

The fix has to break the batch out of "frozen per call." What's the natural unit at which the batch *could* change? A generation isn't one indivisible operation — it's a sequence of decode steps, each one forward pass producing one token per request. The composition of the batch could in principle be reconsidered at *every step*. That's the granularity I want: make scheduling decisions per **iteration** (per forward pass), not per generation. Before each step, decide which requests run *this* step; after the step, retire the ones that just produced their end token and admit waiting ones into the freed slots. The batch is re-formed every iteration. A request that finishes at step 20 is gone at step 21, its slot immediately taken by a queued request. No request ever runs a forward pass over a batch padded with corpses.

Let me make sure this composes with what each request actually needs at a given step, because requests aren't all in the same phase. A fresh request needs prefill (ingest its whole prompt); an in-flight request needs one decode step (one new token). Under per-call batching I think of "the prefill batch" and "the decode batch" as separate phases. But iteration-level scheduling lets me drop that distinction: at each step a request just has some number of tokens it still needs computed — for a fresh request that's its whole prompt, for a decoding request that's the single next token — and the scheduler's job each step is to pick a set of requests and a number of tokens for each, up to a budget, so every request's computed-token count creeps toward its needed count. Prefill and decode become the same kind of work item ("advance request r by n tokens this step"), just with different n. That's a cleaner and more general scheduler, and it's exactly what iteration-level granularity unlocks.

Concretely the scheduler keeps two queues: `waiting` (arrived, not yet started) and `running` (have cache allocated, mid-generation). Each step: walk the `running` requests, and for each, figure out how many new tokens it needs — `num_tokens_with_spec - num_computed_tokens` — clamp it to the remaining per-step token budget, and schedule that many. Decoding requests need ~1 token each; a request still finishing its prompt needs the rest of its prompt. Keep going until the token budget for the step is spent. Then, after the model runs, retire any request that emitted its stop token (free its blocks back to the pool — cheap now, it's just returning block numbers to the free list) and pull from `waiting` to fill the freed capacity. The batch for the *next* step is whatever survived plus whatever was just admitted.

There's a memory-pressure case I have to handle, and it's why paging being in place first matters. With requests joining and leaving every step and each one growing its cache token by token, I can hit a moment where the running set needs one more block than the pool has free. Under contiguous caches this was fatal. Under paging I have a graceful option: **preempt** a request — evict its blocks back to the pool (it can be recomputed or swapped later) — to make room, and put it back on the waiting queue to resume. Preemption is only cheap and safe because freeing is just returning block numbers; the iteration-level scheduler and the block pool are made for each other. So the scheduler, when it can't allocate the blocks a step needs, picks a victim from `running`, preempts it, frees its blocks, and retries — rather than crashing or stalling.

Why does this actually raise throughput at fixed latency, beyond "no padded batches"? Because the average batch occupancy over time goes way up. Under per-call batching, occupancy decays across a batch's life as members finish, and the time-averaged useful batch is a fraction of the peak. Under continuous (iteration-level) batching, every freed slot is refilled immediately, so occupancy stays near the memory ceiling continuously. Since decode is bandwidth-bound and tokens/sec tracks the *live* batch size, holding occupancy at the ceiling instead of letting it decay is a direct throughput multiplier — and it stacks on top of paging, because paging is what raised the ceiling in the first place. It also cuts queueing latency: a waiting request gets admitted at the next step a slot opens, not after a whole batch drains.

The note I'll keep for later: this scheduler is deliberately phrased as "advance each request by some tokens toward its target, within a per-step token budget." That phrasing isn't just elegant — it's the hook that the next refinements plug into. The moment "how many tokens does this request need" and "how do I spend a per-step budget across requests" are first-class, I can start being clever about *which* tokens to compute when (splitting a giant prefill so it doesn't monopolize a step) and *whether* I even need to compute some of them (reusing a shared prefix's cache). But right now the win is removing the temporal waste: stop freezing the batch per call, re-form it every iteration, retire finished requests and admit waiting ones every step.

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

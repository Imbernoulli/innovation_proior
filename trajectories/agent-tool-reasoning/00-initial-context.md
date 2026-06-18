## Research question

Give a frozen LLM a user query and a pile of real RapidAPI tools, and ask it to fulfill the query by
calling tools in the right order with the right arguments. The single thing being designed is the
agent's **search/reasoning policy** — how it explores the action space: which tool to call next, when
to back up after a dead end, when to give up — and the policy trades task success against the number
of LLM queries it spends. Everything else (the tool environment, the answer judge, the model
backbones, the prompts, the decoding parameters) is frozen. The benchmark is StableToolBench
(Guo et al. 2024, arXiv:2403.07714), a stabilized ToolBench (Qin et al. 2023, arXiv:2307.16789) whose
virtual API server replaces flaky live APIs with a cache+simulator and whose GPT-4 judge reports a
Solvable Pass Rate.

## Prior art before the first rung (grounded-acting lineage)

The first rung is a single forward chain over a frozen model; that idea is itself the resolution of a
short lineage, and the fixed harness below is what it converged to.

- **Chain-of-thought prompting (Wei et al. 2022).** Few-shot exemplars that write out intermediate
  reasoning steps before the answer; large models imitate the form and produce their own multi-step
  procedure. Gap: a *closed box* — it reasons over its own pretraining residue with no contact with
  the world, so the moment a step needs a fact the model doesn't hold it invents one and the whole
  chain reasons from the fabrication. For tool use, it even hallucinates APIs that do not exist.
- **Self-consistency (Wang et al. 2022).** Sample many chains, vote on the answer; averages out
  path-level noise. Gap: every sampled chain draws from the same internal well, so it cures *path*
  noise, not *missing grounding* — many chains can march confidently to the same wrong answer.
- **Acting-by-prompting (LLM-as-policy: WebGPT, SayCan, etc., 2021–2022).** Turn each observation into
  text, ask the model for the next action, execute it, feed the new observation back. Gap: real
  contact with the world but no place to *think* between actions, so on a multi-step task it loses the
  goal, repeats a call, and can't synthesize a final answer from what it gathered.
- **Reflect-and-retry (Reflexion, Shinn et al. 2023).** On a failed episode, write a natural-language
  reflection on what went wrong, stash it, and rerun the *whole task* with the reflection prepended.
  Gap: the unit of revision is the entire episode — a fumble at step nine throws away eight good steps
  and re-derives them. It corrects *between* attempts, never *within* one.

## The fixed substrate

A StableToolBench inference harness is frozen and must not be touched. It exposes the agent as a
`CustomSearch` object whose tree is built from a root node carrying the system/user prompt, and it runs
the same `search()` policy across both DeepSeek and Qwen backbones on the I1-instruction subset. The
harness owns the entire ReAct round and hands the policy three helpers, none of which may be modified:

- `self._step(node)` — one LLM call from `node` plus tool execution. It parses the assistant message
  into a `Thought` node and, for a tool call, an `Action` node and an `Action Input` node; it runs the
  call against the virtual API server, sets `observation_code` (0 ok, 1 hallucinated API name, 3 final
  answer, 4 give up), flips `is_terminal`/`pruned`, increments `self.query_count`, and returns the new
  leaf(s). Returns `[]` once the query budget is hit.
- `self._add_diversity_prompt(node)` — if `node` already has children, appends a temporary prompt
  listing the actions already tried from `node` and asking for a *different* one; returns `True` so the
  caller marks that message invalid afterward (it is generation scaffolding, not part of the child's
  history).
- `self._rank_nodes(candidates)` — LLM **pairwise** ranking of candidate nodes via `sum_based_rankn`;
  returns a per-candidate score (higher is better) and charges the comparison calls to `query_count`.

Tree/node state is read-only-by-convention but the policy may *attach* its own attributes to nodes
(as the baselines attach `prior_score`): `self.query_count`/`self.max_query_count`,
`self.terminal_node`/`self.give_up_node`, `self.status` (set to 1 on success), `self.answer_count`,
`self.single_chain_max_step` (depth cap, in raw nodes — a logical round is Thought+Action+Action Input,
so three), `self.tree_beam_size` (children per expansion); and `node.is_terminal`, `node.pruned`,
`node.observation_code`, `node.get_depth()`, `node.children`, `node.father`, `node.messages`,
`node.make_finish(k)`.

## The editable interface

Exactly one region is editable — the `search(self, root_node)` method of `CustomSearch` in
`custom_search.py` (lines 368–439). Every rung is a fill of this one method (plus any private helper
methods it calls). The starting point is the scaffold default: a **greedy chain** — follow one path,
no backtracking.

```python
    # EDITABLE region of custom_search.py — default fill (greedy chain, no backtracking)
    def search(self, root_node):
        """Core search logic. Modify this method to implement your strategy.

        Helpers (do NOT modify, just call): self._step(node), self._add_diversity_prompt(node),
        self._rank_nodes(candidates). State: self.query_count, self.max_query_count,
        self.terminal_node, self.give_up_node, self.status, self.answer_count,
        self.single_chain_max_step, self.tree_beam_size. Node: node.is_terminal, node.pruned,
        node.observation_code, node.get_depth(), node.children, node.father.
        """
        now_node = root_node
        for step in range(self.single_chain_max_step):
            if self.query_count >= self.max_query_count:        # budget exhausted
                break
            if len(self.terminal_node) >= self.answer_count:    # already have enough answers
                break

            new_leaves = self._step(now_node)                   # one LLM call + tool exec
            if not new_leaves:
                break
            now_node = new_leaves[-1]

            if now_node.is_terminal:                            # produced a final answer
                self.status = 1
                self.terminal_node.append(now_node)
                break
            if now_node.pruned:                                 # dead end / give up
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                break
            if now_node.get_depth() >= self.single_chain_max_step:
                now_node.pruned = True
                break
```

## Evaluation settings

The same `search()` policy is run across multiple agent backbones on the **I1-instruction** subset of
StableToolBench (50 scored queries per backbone), seed 42. Three diagnostics are reported per backbone:
`pass_rate` — fraction of queries with a valid final answer (higher is better, the primary signal,
alongside the GPT-4-judge Solvable Pass Rate `sopr`); `avg_queries` — average LLM queries per task
(lower is better, the efficiency signal); and `give_up_rate` — fraction of queries the agent abandons
(lower is better, a diagnostic). The score emphasizes answer quality; query count and give-up rate are
efficiency and diagnostic signals.

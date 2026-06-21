## Research question

A frozen LLM is given a user query and a set of real RapidAPI tools. It must fulfill the query by calling the right tools in the right order with the right arguments. The only design freedom is the agent's **search/reasoning policy** — how it explores the action space, chooses the next tool call, recognizes a dead end, and decides when to stop — trading task success against the number of LLM queries spent. The environment, answer judge, model backbones, prompts, and decoding parameters are all fixed. The benchmark is StableToolBench (Guo et al. 2024; built on ToolBench, Qin et al. 2023), which provides a virtual API server and a GPT-4 judge that reports a Solvable Pass Rate.

## Prior art / Background / Baselines

- **Chain-of-thought prompting (Wei et al. 2022).** Few-shot exemplars lead the model to write intermediate reasoning steps before the final answer. Gap: it is a closed box that never queries the world, so whenever a step needs a fact the model does not know, it fabricates it; for tool use, it can invent APIs that do not exist.
- **Self-consistency (Wang et al. 2022).** Sample many reasoning chains and take a majority vote. Gap: every chain samples from the same model-internal distribution, so it reduces path-level noise but not missing grounding; many chains can converge on the same wrong answer.
- **Acting-by-prompting / LLM-as-policy (WebGPT, SayCan, etc., 2021–2022).** At each step the model observes the world, emits an action, and the action is executed before the next observation is fed back. Gap: there is no explicit reasoning between actions, so on multi-step tasks the model loses the goal, repeats calls, and cannot reliably synthesize a final answer from gathered observations.
- **Reflect-and-retry (Reflexion, Shinn et al. 2023).** After a failed episode, generate a natural-language reflection, store it, and rerun the entire task with the reflection prepended. Gap: correction happens only between whole episodes, so one mistake forces the agent to discard all earlier steps and reconstruct them.

## Fixed substrate / Code framework

The StableToolBench inference harness is frozen. The agent is a `CustomSearch` object whose tree starts from a root node carrying the system/user prompt; the harness runs the same `search()` policy across DeepSeek and Qwen backbones on the I1-instruction subset. The harness owns the full ReAct round and exposes three helpers that must not be modified:

- `self._step(node)` — one LLM call from `node` plus tool execution. It parses the assistant message into `Thought`, `Action`, and `Action Input`; runs the call against the virtual API server; sets `observation_code` (0 ok, 1 hallucinated API name, 3 final answer, 4 give up) and `is_terminal`/`pruned`; increments `self.query_count`; and returns the new leaf(s). Returns `[]` once the query budget is hit.
- `self._add_diversity_prompt(node)` — if `node` already has children, appends a temporary prompt listing actions already tried from `node` and asking for a different one; returns `True` so the caller can mark that message invalid afterward.
- `self._rank_nodes(candidates)` — LLM **pairwise** ranking of candidate nodes via `sum_based_rankn`; returns a per-candidate score (higher is better) and charges the comparison calls to `query_count`.

Policy-relevant state: `self.query_count`/`self.max_query_count`, `self.terminal_node`/`self.give_up_node`, `self.status` (set to 1 on success), `self.answer_count`, `self.single_chain_max_step` (depth cap in raw nodes; a logical round is Thought+Action+Action Input, so three), `self.tree_beam_size` (children per expansion); and per node `is_terminal`, `pruned`, `observation_code`, `get_depth()`, `children`, `father`, `messages`, `make_finish(k)`. The policy may attach its own attributes to nodes.

## Editable interface

Exactly one region is editable — the `search(self, root_node)` method of `CustomSearch` in `custom_search.py` (lines 368–439). Every rung is a fill of this one method (plus any private helper methods it calls). The starting point is the scaffold default below: a **greedy chain** that follows one path and never backtracks.

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

The same `search()` policy is run across multiple agent backbones on the **I1-instruction** subset of StableToolBench (50 scored queries per backbone), seed 42. Four diagnostics are reported per backbone: `pass_rate` — fraction of queries with a valid final answer (higher is better; the primary quality signal, alongside the GPT-4-judge Solvable Pass Rate `sopr`); `avg_queries` — average LLM queries per task (lower is better; the efficiency signal); and `give_up_rate` — fraction of queries the agent abandons (lower is better; a diagnostic signal).

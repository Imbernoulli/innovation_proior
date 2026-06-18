**Problem.** Fulfill a multi-tool instruction with a frozen LLM in a ReAct loop (think → call API →
read real response → think), under a hard query budget. Pure reasoning hallucinates tools and facts;
pure acting can't think between calls. The floor is the simplest control flow over the harness's
`_step`: a single forward chain.

**Key idea.** Interleave thought and action in one trajectory over an augmented action space
Â = A ∪ L (a thought has no environment effect, just updates the context). The `search()` policy is
*only* the loop deciding when to call `_step` and when to stop — the parser, tool execution, status
codes, and terminal detection all live inside the frozen `_step`. The greedy chain follows exactly one
path: step, move to the leaf, stop on a final answer, a dead end, or the depth/budget cap. No
backtracking, no diversity prompt, no ranking — so it spends the minimum possible queries.

**Step-1 edit.** The literal default fill of the editable region (`custom_search.py`, lines 368–439),
equivalent to CoT@1 / single-chain ReAct.

**Why it is the floor.** A single chain cannot retract a committed action: an early wrong call
propagates into a faulty loop, and one linear pass explores only one route through an unbounded action
space. It is cheapest (lowest `avg_queries`) and weakest (lowest `pass_rate`) by construction.

**What to watch.** Lowest `avg_queries` on every backbone (the symptom, not a virtue — it quits early);
weakest `pass_rate`, with the gap widening on harder instructions and weaker backbones; non-trivial
`give_up_rate`. That structural failure forces a *tree* with retraction at step 2.

```python
    def search(self, root_node):
        """Greedy chain: follow one path, no backtracking."""
        now_node = root_node
        for step in range(self.single_chain_max_step):
            if self.query_count >= self.max_query_count:
                break
            if len(self.terminal_node) >= self.answer_count:
                break

            new_leaves = self._step(now_node)
            if not new_leaves:
                break

            now_node = new_leaves[-1]

            if now_node.is_terminal:
                self.status = 1
                self.terminal_node.append(now_node)
                break

            if now_node.pruned:
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                break

            if now_node.get_depth() >= self.single_chain_max_step:
                now_node.pruned = True
                break
```

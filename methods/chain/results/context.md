# Context: scalable collaboration among LLM workers

## Research question

A single large language model prompted once produces one artifact: an answer, a program, a
plan, or a piece of writing. The artifact inherits whatever omissions, local mistakes, and
hallucinations the model committed on that first pass. Reflection-and-revision methods show
that a draft can improve when it is inspected and rewritten, and role-based agent systems show
that separate workers can bring different perspectives to the same task. The unresolved
question is how to make that collaboration scale beyond a small, hand-built workflow.

The target is an inference-time collaboration recipe with four requirements. It should let the
experimenter vary the number of workers without redesigning the whole system. It should describe
who hands work to whom using a compact object rather than task-specific prose. It should always
terminate, with a clear final artifact. It should also keep the visible context for each worker
under control: if later workers must read every intermediate conversation, token cost and
latency grow too quickly for large teams. Existing systems supply pieces of this picture, but
none supplies a general, scalable recipe for ordered handoffs among many workers.

## Background

Large language models already improve with scale during training: larger models, datasets, and
compute budgets predictably lower loss and can unlock new abilities (Vaswani et al. 2017;
Brown et al. 2020; Kaplan et al. 2020; Wei et al. 2022b). That training-time story motivates a
separate inference-time question: once a capable foundation model exists, can multiple
instances of it collaborate in a way that improves the artifact without retraining?

Three pre-existing ideas matter. First, iterative refinement treats an artifact as a draft: a
system generates, reviews, and revises until a stopping rule is met. Second, role prompting
turns foundation models into task-specific workers, such as a planner, reviewer, coder, or
tester. Third, software-style workflows already pass artifacts between stages, keeping the
main deliverable while discarding much of the local discussion that produced it. The tension is
that these ideas are usually implemented in bespoke workflows. A useful scaling experiment
needs a reusable handoff description and a run procedure that does not depend on the task being
software development.

The central engineering constraint is context. The simplest many-worker design is to broadcast
all messages to everyone, but then the most burdened worker sees an accumulated transcript whose
size grows with the amount of interaction in the whole team. Context-window limits, token cost,
and latency become the bottleneck before the number of workers becomes scientifically
interesting. A scalable system must preserve enough information to continue the task while
avoiding full-transcript broadcast.

## Baselines

**Self-Refine (Madaan et al. 2023).** A single model improves its own output by alternating
feedback and refinement. Given input `x`, it generates `y0 = M(p_gen || x)`, then loops
`fb_t = M(p_fb || x || y_t)` and `y_{t+1} = M(p_refine || x || y_t || fb_t)` until a stop
condition. The method needs no retraining or extra model. Its limitation is that one model is
writer, reviewer, and reviser at once, so the review is constrained by the same perspective
that produced the draft.

**CAMEL (Li et al. 2023).** Two role-played agents cooperate through inception prompting: an
AI user gives instructions and an AI assistant follows them. The role split is important
because single-agent role play can suffer role flipping, instruction repetition, and fake or
rubber-stamp replies. Its limitation is scale and composition: it is a dyad, not a recipe for
coordinating many workers.

**ChatDev (Qian et al. 2023).** Software development is decomposed into a fixed chat chain of
phases such as design, coding, and testing. Each subtask is handled by an instructor/assistant
dialogue, and the system distinguishes local dialogue history from the solution passed to later
phases. Its limitation is rigidity: the structure is the software waterfall, hand-written for a
small set of phases, so it does not provide a general way to vary collaboration shape or team
size.

**MetaGPT (Hong et al. 2023).** Human standardized operating procedures are encoded into an
assembly line of role agents such as product manager, architect, and engineer. Structured
documents pass through the line. Its limitation is the same kind of workflow specificity: the
procedure and roles are engineered for a particular process rather than generated from a small
set of structural knobs.

**GPTSwarm (Zhuge et al. 2024).** A swarm is represented as a computational graph whose nodes
are customized operations and whose edges carry information; prompts and connectivity can be
optimized during reasoning. Its limitation is the customization burden: node functions and
connections must be adapted heavily to the task, which makes cheap, heterogeneous scaling hard
to study.

**AgentVerse (Chen et al. 2024).** Expert agents are assembled dynamically for multi-agent
linguistic interaction, often with a coordinator-like organization. Its limitation is that a
central organizer can become a context bottleneck, and the resulting interaction pattern still
does not give a simple, reusable way to sweep team size and structure.

**Independent sampling ensembles.** Majority voting and best-of-N sampling equalize compute by
running many non-interacting generations and aggregating the outputs. Their limitation is that
the samples do not refine one another, so they test extra sampling rather than collaboration.

## Evaluation settings

Natural yardsticks span closed-domain reasoning, code, software engineering, and open-ended
generation:

- **MMLU** (Hendrycks et al. 2020): multiple-choice questions across subjects, scored by
  accuracy.
- **HumanEval** (Chen et al. 2021): function-level code generation, scored by pass@k against
  hidden unit tests.
- **SRDD**: repository-level software development from textual requirements, scored by
  completeness, executability, and consistency.
- **CommonGen-Hard**: sentence generation from discrete concepts, scored by grammar, fluency,
  relevance, and logical consistency.

A practical protocol uses a fixed foundation model for all workers, a small default worker
count aligned with existing multi-agent baselines, and a fixed cap on exchange rounds inside a
single local interaction. The worker count is the primary knob to sweep when testing whether
collaboration continues to help as teams grow.

## Code framework

The harness has three empty slots: a handoff recipe, the local update made by a worker when it
receives an earlier artifact, and the run loop that applies the recipe until a final artifact is
available.

```python
from typing import Callable, Dict, List, Tuple


def choose_handoffs(worker_count: int) -> List[Tuple[int, int]]:
    """Return ordered pairs (sender, receiver) for workers 0..worker_count-1."""
    # TODO: the handoff recipe we will design.
    raise NotImplementedError


class Worker:
    """A foundation-model worker with local state."""

    def __init__(self, worker_id: int, model):
        self.id = worker_id
        self.model = model
        self.received: Dict[int, str] = {}
        self.artifact = ""

    def update(self, task: str, incoming: str) -> str:
        # TODO: how to turn an incoming artifact into the worker's artifact.
        raise NotImplementedError


class CollaborationRun:
    """Owns the workers and applies the handoff recipe."""

    def __init__(
        self,
        handoffs: List[Tuple[int, int]],
        worker_count: int,
        make_worker: Callable[[int], Worker],
    ):
        self.workers = {i: make_worker(i) for i in range(worker_count)}
        self.handoffs = list(handoffs)

    def execute(self, task: str) -> str:
        # TODO: apply the handoffs and return the final artifact.
        raise NotImplementedError
```

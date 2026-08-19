# traj-build — convert a ladder-arc method into a trajectory unit

Repo /srv/home/bohanlyu/innovation_proior, branch main. You get SLUG (a method under methods/<SLUG>/) and TRAJ_NAME (target dir under trajectories/). Reason it qualifies: its record documents a genuine improvement ladder — successive solutions to ONE research question, each with real measured feedback.

## Format — copy the existing convention EXACTLY
Study trajectories/agent-tool-reasoning/ first (meta.json, 00-initial-context.md, NN-<slug>-{reasoning,answer,train_answer}.md, NN-feedback.md). Your unit: trajectories/<TRAJ_NAME>/ with the same shapes. Do NOT create agentic_messages.json (separate later pass). Do NOT edit trajectories.json (site registration awaits review).

## Sources
methods/<SLUG>/results/*, notes/*, refs/, src/ hold the record (papers, ablation tables, author code history, self-accounts). Every feedback number MUST exist in those sources (or in a source you fetch and save to methods/<SLUG>/refs/ with a notes/sources.md entry). NEVER invent numbers. Quote-check each number before writing it.

## Content rules (the epistemics are the whole point)
- 00-initial-context.md: the research question + starting state + evaluation protocol (metrics, datasets) + any PRE-DATING known facts, as givens. Contemporaneous "what we have now" voice; no foreshadowing.
- Rung n reasoning (NN-<rungslug>-reasoning.md): PROPOSAL voice. It may use everything observed in feedback ≤ n-1 (quote those numbers freely — they are observations already received) plus on-page computation. It must NOT state the outcome of its own rung-n proposal. It ends by committing to the rung-n variant and what the test will decide.
- Rung n answer / train_answer: the concrete deliverable of that rung (method description + code/config where applicable), still proposal voice for rung n.
- NN-feedback.md: the environment speaks — the REAL measured numbers for rung n from the record, stated plainly (metric table + one-line factual notes, no coaching, no hints about what to do next).
- Final rung = the published method. The ladder must be the DOCUMENTED one (e.g. convnext: ResNet-50 baseline 76.1 → modern training recipe 78.8 → patchify/stage-ratio → depthwise+width 80.5 → kernel sweep 79.9/80.4/80.6 → micro-design → final). 3-8 rungs; merge micro-steps if the record only gives grouped numbers.
- No provenance leaks in any file ("the paper", "the authors", arXiv ids, "as we now know"). In-frame year conditioning: the narrator knows only what existed then + prior feedback.
- reasoning per rung: substantial (aim 800-2000 words), deriving WHY this rung's change is the right next move given the last feedback — competing options, why this one, predictions. No filler.

## Procedure
1. Read the template trajectory. 2. Mine the ladder from sources; write a rung table (variant, motivation, numbers, source file+line) into trajectories/<TRAJ_NAME>/notes-ladder.md first. 3. Write all files. 4. Self-check: every feedback number greppable in a source file; no rung reasoning contains its own rung's results; lint_inframe clean for the new dir (run `python3 tools/lint_inframe.py | grep <TRAJ_NAME>`). 5. ONE commit: `git add trajectories/<TRAJ_NAME> && git commit -m "traj-build: <TRAJ_NAME> — <n> rungs from documented ladder (<SLUG>)"`. Retry on index.lock.

## Output (JSON only)
{"slug":"...","traj_name":"...","rungs":<int>,"numbers_verified":<int>,"sources_used":["..."],"commit":"<sha>","gaps":"<≤160 chars: what the record lacks>"}

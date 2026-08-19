# kvtuner4-pertoken-qwen25-3b changelog

## 2026-08-18 — epistemic correction (svfix self-account grounding reverted to hypothesis voice)
- `results/reasoning.md` (layer-axis paragraph, "So I have the key-versus-value axis..."): an
  earlier svfix pass ("ICML slides self-account") had grounded the claim "the layer axis I can't
  derive, must be observed" in a concrete reported result — a K8V2-vs-K4V4 attention-output-error
  ranking flip between layer 0 and layer 1 on Qwen2.5-7B under math generation, plus a claim that
  the same reversal "recurs in several of the sensitive layers on Qwen2.5-3B and -7B specifically."
  That is a self-supplied observation: the narrator, mid-proposal, states the outcome of the
  method's own per-layer sensitivity measurement before the method's own experiments have happened
  in the frame. Reworded to a hypothesis/discriminating-design/prediction/decision-rule form: does
  the *ranking* of two candidate pairs at matched bit budget hold across adjacent layers, or can it
  flip layer to layer the way a smooth depth heuristic could never reproduce — and if even one such
  flip is possible anywhere in the network, that's reason enough not to skip measuring every layer
  directly. No numbers or claimed outcomes remain; the decision rule (why the layer axis must be
  searched, not derived from depth) is unchanged and still carries the paragraph's landing. The
  general pre-existing claims in the same paragraph ("stable across prompts," "no clean depth
  heuristic," "Qwen2.5-7B and Qwen2.5-Math break already at INT4 key") predate this svfix pass and
  are out of scope here.
- `answer.md` / `train_answer.md`: untouched by the flagged svfix commit — no corresponding text to
  fix.
- Landing unaffected: the paragraph's conclusion ("I have to take it as observed... decide the
  per-layer precision once, before deployment") still follows from the pre-existing "no clean depth
  heuristic, interleaved sensitivity" framing plus the now-hypothetical ranking-instability argument,
  so this method does not need trajectory-conversion queuing.

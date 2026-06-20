Measured results for the SFT model, against the midtrained model. Accuracies higher-is-better; ChatCORE is the centered-mean summary.

**Report-card MID → SFT ($100 / depth-20 reference run):**

| Metric | MID | SFT | Δ |
|---|---|---|---|
| ARC-Easy | 0.3561 | 0.3876 | +0.0315 |
| ARC-Challenge | 0.2875 | 0.2807 | −0.0068 |
| MMLU | 0.3111 | 0.3151 | +0.0040 |
| GSM8K | 0.0250 | 0.0455 | +0.0205 |
| HumanEval | 0.0671 | 0.0854 | +0.0183 |
| ChatCORE | 0.0730 | 0.0884 | +0.0154 |

Reading: SFT lifts the summary ChatCORE from 0.073 to **0.0884**. The biggest single gain is ARC-Easy (+0.032) and the most important targeted gain is GSM8K, which nearly doubles (0.025 → 0.0455) from the extra tool-use epochs — though at 0.045 it is still the weakest row, exactly the imitation ceiling the finale RL attacks. HumanEval climbs to 0.085. ARC-Challenge dips slightly (−0.007), within the noise of a tiny model on the hardest split; MMLU is roughly flat. The mixture and weighting buy broad, real improvement across the assistant's behavior.

Provenance: the repo author's published depth-20 report-card SFT values (original nanochat "$100 speedrun" announcement). The SFT mixture and the warm-start / best-fit-pack logic are committed verbatim in `scripts/chat_sft.py` (with `--mmlu-epochs 3`, `--gsm8k-epochs 4` defaults), and the ChatCORE metric in `report.py`. The per-stage numbers themselves are not committed as a file at this repo HEAD; not re-run by us.

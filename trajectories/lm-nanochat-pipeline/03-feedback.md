Measured results for the midtrained model — the first stage where the chat report card is defined (the base model couldn't play any of these games). Accuracies are higher-is-better; ChatCORE is the centered-mean summary (0 = random, 1 = perfect).

**Report-card MID column ($100 / depth-20 reference run):**

| Metric | MID |
|---|---|
| ARC-Easy | 0.3561 |
| ARC-Challenge | 0.2875 |
| MMLU | 0.3111 |
| GSM8K | 0.0250 |
| HumanEval | 0.0671 |
| ChatCORE | 0.0730 |

Reading: the model now plays every game. Multiple choice is clearly above the 0.25 random floor (ARC-Easy 0.356, MMLU 0.311; ARC-Challenge 0.288 is the hardest). GSM8K is just off the floor (0.025) — the tool-use *format* is installed but the model rarely lands the full multi-step solution yet. HumanEval 0.067 confirms the Python channel works end-to-end. ChatCORE 0.073 is the single positive summary: the assistant is real but rough.

Provenance: these are the repo author's published depth-20 report-card MID values (the original nanochat "$100 speedrun" announcement). They are not committed as a file at this repo HEAD — the current `runs/speedrun.sh` folds midtraining and SFT into a single SFT stage, but `nanochat/report.py`'s `chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]` and the centered-mean ChatCORE definition are committed and generate exactly this row. Not re-run by us.

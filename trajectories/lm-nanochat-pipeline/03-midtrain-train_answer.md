The base model clears the substrate test — CORE 0.2219 at depth-20, low and smooth bpb, the same architecture+optimizer clearing GPT-2's 0.256525 in the d24–d26 range on one node — but behaviorally it is useless as an assistant. It completes documents. Hand it "Why is the sky blue?" and it will happily continue the *text*, maybe inventing a quiz with three more questions, because in its training data a question is just a string that precedes more strings. It has never seen the *structure* of a conversation — that a "user" turn ends and an "assistant" turn begins and then stops — has no idea that lettered options want a single letter back, and no idea it is even *allowed* to call a tool instead of doing arithmetic in its head. None of these are knowledge gaps I can close by training longer on web text; they are *format and behavior* gaps. So before any polished finetuning I want a stage that bridges the base model from raw-text completion into the conversational, multiple-choice, tool-using world while it still has plenty of plasticity.

I propose **midtraining**: a format/distribution adaptation that trains the still-plastic base model on a mixture of rendered conversations, multiple-choice questions, and tool-annotated math, under the same next-token cross-entropy objective but masked to the assistant's tokens. The first thing to install is the conversation format itself, where the special tokens reserved in the tokenizer finally earn their place. A conversation is a structured object — a list of messages, alternating user and assistant — and I render it into the one flat token stream the model consumes in a way it can learn to parse and to *generate the boundaries of*. The scheme begins each document with `<|bos|>`, then wraps every message in role delimiters: a user turn is `<|user_start|>` … text … `<|user_end|>`, an assistant turn is `<|assistant_start|>` … text … `<|assistant_end|>`. The model never saw these tokens in pretraining (they were inert), so midtraining is where they acquire meaning — it learns that `<|assistant_start|>` is its cue to begin responding and `<|assistant_end|>` is where it must stop. That stop token is not a detail: without a learned end-of-turn an assistant rambles forever, and `<|assistant_end|>` is exactly what lets generation terminate cleanly at inference.

The asymmetry that defines this whole family of stages is that I do **not** train on everything in the rendered stream. The user's tokens are given to the model at test time, so it should never be in the business of predicting what the user will say. Concretely the renderer returns, alongside the token ids, a parallel mask: 1 on the tokens the assistant is responsible for producing — its message content and its closing `<|assistant_end|>` — and 0 on the BOS, the user turn, and the role-delimiter scaffolding it does not author. At training time the masked positions get the ignore-index in the cross-entropy, so the gradient flows only from "what the assistant should say next." This is what teaches *response behavior* rather than *transcript memorization*.

Then the multiple-choice protocol, because two eval tasks (ARC, MMLU) and a large chunk of useful knowledge come as a question plus lettered options wanting one letter back. The base model has the latent knowledge but no notion of the protocol — that the expected output is exactly "A" or "C", not a paragraph. The fix is data: render a pile of multiple-choice questions as conversations (question and options as the user turn, the correct letter as the assistant turn) and mix them in, drawing on a large auxiliary multiple-choice set (MMLU's auxiliary train split, ~100K rows), so the model learns the motion "read options → emit the single correct letter." The eval harness then reads the model's answer the natural way, letting it commit to a letter, and the model now knows that is the game.

The piece I am most interested in is **tool use** — a calculator. A 20-layer model is hopeless at multi-step arithmetic in its head; GSM8K word problems chain several exact multiplications and additions, and a tiny model fumbles the digits even when its plan is right. But it does not *have* to do arithmetic in its head if I give it a tool. The reserved `<|python_start|>`/`<|python_end|>` tokens are a channel: when the assistant wants to compute it emits `<|python_start|>`, writes a Python expression, emits `<|python_end|>`, and the *runtime* — not the model — executes that expression in a sandbox and feeds the result back inside `<|output_start|>`/`<|output_end|>`; the model then continues its reasoning using the returned number. GSM8K is the perfect teacher because its reference solutions already carry calculator annotations — every arithmetic step is written `<<12/60=0.2>>`, the expression and its result — so I parse them: the expression `12/60` becomes a `python` part rendered between `<|python_start|>`/`<|python_end|>`, and the result `0.2` becomes a `python_output` part rendered between `<|output_start|>`/`<|output_end|>`. Mixing GSM8K-as-conversations in teaches the *motion* of tool use: recognize when arithmetic is needed, hand it off, consume the result, keep going.

The masking subtlety on tool use is the difference between a model that learns to *use* a tool and one that learns to *hallucinate the tool's answers*. The assistant's *invocation* — `<|python_start|>`, the expression, `<|python_end|>` — is supervised (mask 1): the model should learn to write the call. But the tool's *output* — `<|output_start|>`, the returned number, `<|output_end|>` — is masked out (mask 0), because at test time those tokens come from the actual Python interpreter, not the model. If I trained the model to predict the output tokens I would be teaching it to fabricate calculator results, the exact failure mode tool use exists to prevent. So I supervise the call and never the result; the renderer's per-part logic encodes precisely this — text and `python` parts get mask 1, `python_output` parts get mask 0. The same `<|python_start|>`/`<|python_end|>` channel plus the sandboxed executor also covers HumanEval, where the model writes short Python and the runtime runs and checks it, so installing the tool format here sets up both the math and the coding evals.

What midtraining really *is*, then, is a domain/format adaptation: take the base model with all its plasticity and train it on (a) rendered conversations so it learns the user/assistant protocol and the stop token, (b) multiple-choice questions so it learns to emit a single letter, and (c) tool-annotated math so it learns the Python-REPL invocation pattern. The objective is unchanged next-token cross-entropy, now masked to the assistant's tokens, and the data distribution is the conversational/tool-use world rather than raw web text. The training loop is essentially the pretraining loop — the same MuonAdamW optimizer, the same precision — with the conversation-rendering dataloader swapped in, and because conversations are short relative to web documents, the packing wastes far fewer tokens. This is the stage where the chat report card *first lights up*: before it, ARC/MMLU/GSM8K/HumanEval/ChatCORE are undefined because the model cannot even play the games; after it, I get the first real numbers, with ChatCORE positive (above random on the centered mean). The responses will be format-correct but rough — midtraining is a broad, coarse adaptation on a heterogeneous mixture, brittle on conversational polish and on holding the protocol under varied phrasing — and tightening that, with a cleaner and better-weighted instruction mixture, is the next stage.

```python
def render_conversation(self, conversation, max_tokens=2048):
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int): token_ids = [token_ids]
        ids.extend(token_ids); mask.extend([mask_val] * len(token_ids))
    bos = self.get_bos_token_id()
    user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
    assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
    python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
    output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")
    add_tokens(bos, 0)
    for i, message in enumerate(messages):
        content = message["content"]
        if message["role"] == "user":                         # user turn: never supervised (mask 0)
            add_tokens(user_start, 0); add_tokens(self.encode(content), 0); add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)
            if isinstance(content, str):
                add_tokens(self.encode(content), 1)           # assistant text: supervised (mask 1)
            elif isinstance(content, list):
                for part in content:
                    value_ids = self.encode(part["text"])
                    if part["type"] == "text":
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":            # tool CALL: supervised (mask 1)
                        add_tokens(python_start, 1); add_tokens(value_ids, 1); add_tokens(python_end, 1)
                    elif part["type"] == "python_output":     # tool OUTPUT: NOT supervised (mask 0)
                        add_tokens(output_start, 0); add_tokens(value_ids, 0); add_tokens(output_end, 0)
            add_tokens(assistant_end, 1)                       # the stop token IS supervised
    return ids[:max_tokens], mask[:max_tokens]
```

```python
# GSM8K reference solutions carry calculator annotations <<expr=result>>; parse into tool parts:
parts = re.split(r'(<<[^>]+>>)', answer)
for part in parts:
    if part.startswith('<<') and part.endswith('>>'):
        expr, result = part[2:-2].rsplit('=', 1)
        assistant_message_parts.append({"type": "python", "text": expr})         # the call (supervised)
        assistant_message_parts.append({"type": "python_output", "text": result})# the result (masked)
    else:
        assistant_message_parts.append({"type": "text", "text": part})
```

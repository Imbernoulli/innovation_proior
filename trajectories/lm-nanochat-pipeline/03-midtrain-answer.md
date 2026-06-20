**Problem (from step 2).** The base model only completes text — it has no notion of a conversation (no user/assistant turns, no stop token), no multiple-choice protocol (emit one letter), and no tool channel (it does arithmetic in its head and fails). These are format/behavior gaps, not knowledge gaps; web-text training can't fix them. Before polished finetuning, adapt the still-plastic base model into the conversational, multiple-choice, tool-using world.

**Key idea.** Midtraining = format/distribution adaptation on a mixture, using the special tokens reserved in the tokenizer:
- **Conversation format:** render each message wrapped in role delimiters (`<|user_start|>…<|user_end|>`, `<|assistant_start|>…<|assistant_end|>`); the model learns `<|assistant_start|>` as its cue to respond and `<|assistant_end|>` as where to stop.
- **Loss only on the assistant's tokens:** a parallel mask is 1 on the assistant's content + its closing `<|assistant_end|>`, 0 on BOS, the user turn, and the scaffolding — so it learns *response behavior*, not transcript memorization.
- **Multiple-choice data** (MMLU auxiliary train) teaches "read options → emit one letter."
- **Tool use:** the `<|python_start|>…<|python_end|>` channel lets the assistant write a Python expression that the *runtime* executes, returning the result inside `<|output_start|>…<|output_end|>`. GSM8K's `<<expr=result>>` calculator annotations are parsed into `python` / `python_output` parts. Crucially the assistant's *call* is supervised (mask 1) but the tool's *output* is masked (mask 0) — never train the model to fabricate the calculator's answer.

**Why it works.** The model has the latent knowledge; what it lacks is the protocol. Installing the format with masked next-token loss teaches it the turn structure, the stop token, the single-letter answer convention, and the motion of tool use — recognize when arithmetic is needed, hand it to the tool, consume the result, continue. This is the stage where the chat report card first becomes measurable (the base model couldn't play any of the games).

**Change / code.** The conversation renderer and its supervision mask (assistant call supervised, tool output not):

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

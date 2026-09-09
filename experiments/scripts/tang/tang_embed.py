#!/usr/bin/env python3
"""Embed annotated documents with Qwen3-Embedding-4B (paper S1.3): text = 'Questions addressed' + ' ' + 'Method'
from the scholarly annotation; L2-normalized last-token pooling, no instruction (symmetric doc-doc use).
Input: one or more annotation jsonl files. Output: .npy matrix + ids.json."""
import argparse, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
ap = argparse.ArgumentParser(); ap.add_argument("--ann", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max-len", type=int, default=1024)
a = ap.parse_args()
ids, texts = [], []
for fn in a.ann:
    for l in open(fn):
        r = json.loads(l)
        if r.get("unparsed"): continue
        an = r["analysis"]; ids.append(r["doc_id"]); texts.append((an["Questions addressed"] + " " + an["Method"]).strip())
print("docs", len(texts), flush=True)
tok = AutoTokenizer.from_pretrained(a.model, padding_side="left"); model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
order = np.argsort([-len(t) for t in texts])
with torch.no_grad():
    for i in range(0, len(texts), a.bs):
        idx = order[i:i+a.bs]; batch = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to("cuda")
        h = model(**batch).last_hidden_state[:, -1]           # left padding -> last token is the real last token
        e = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy(); out[idx] = e
        if (i // a.bs) % 50 == 0: print(i, flush=True)
np.save(a.out + ".npy", out); json.dump(ids, open(a.out + ".ids.json", "w")); print("DONE", out.shape)

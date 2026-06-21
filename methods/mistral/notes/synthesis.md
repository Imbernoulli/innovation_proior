# Mistral 7B Synthesis

This legacy synthesis note has been refreshed to point at the strict evidence bundle created for the 2026-06-18 audit. The authoritative reconstruction notes are:

- `notes/source_matrix.md`
- `notes/discovery_synthesis.md`

Key audited facts:

- Primary source: arXiv 2310.06825 source under `refs/primary/`.
- Canonical code: `code/mistral-src` at HEAD `9eaeb91c17450e09021b6065a1d5cc69876507c8`, especially `src/mistral_inference/transformer_layers.py` and `src/mistral_inference/cache.py`.
- GQA factor: `n_heads / n_kv_heads = 32 / 8 = 4`, so KV cache width and bandwidth are reduced 4x.
- Sliding-window span: the operational mask keeps at most `W` keys per query. The source describes information moving up to `W` positions per layer; with 32 layers and `W=4096`, the theoretical transitive span is `131072` tokens.
- Rolling cache: absolute position `i` writes to slot `i mod W`; old ring order is recovered with `unrotate`; at 32K tokens and `W=4096`, cache length is reduced 8x.
- Prefill/chunking: first prefill, subsequent prompt chunks, and one-token decode use distinct masks in `BufferCache._get_input_metadata_layer`.

Benchmark tables, instruction tuning, chat/system-prompt examples, and leaderboard claims remain out of scope for the reasoning deliverables.

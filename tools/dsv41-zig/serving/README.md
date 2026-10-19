# Serving a DeepSeek-V4.1-Flash-1017 EXL3 pack on the TP2 engine

The September packs (`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`, our banked 2.9 bpw) already carry the engine's
names and dtypes. The 1017 packs differ in two ways the loader handles, and one way this folder prepares:

1. **MTP markov head names.** The 1017 converter writes `mtp.N.markov_head.markov_w1.weight` /
   `markov_w2.weight`; the engine reads `mtp.N.markov_head.embed.weight` / `head.weight`. `checkpoint.zig` falls
   back to the converter's names when the engine's own are absent, so a raw 1017 pack loads as it is. The Python
   line's rank-cache writer (`deepseek-v41-tp2`, `weights.py`) only looks up the engine's names, so a pack served
   through that writer needs the rename shim below.
2. **Engram q / k in bf16.** The 1017 packs hold `layers.N.engram.q_weight` / `k_weight` in bf16; the engine reads
   them as f32. `checkpoint.zig` widens bf16 -> f32 exactly (bf16 is an f32's top 16 bits), and the Python line's
   `.to(torch.float32)` does the same. No shim needed.
3. **The MTP markov head's dtype.** The 1017 checkpoint holds `mtp.2.markov_head.head.weight` in bf16; some community
   packs keep it, our converter casts it to f16. The kernel reads fp16, so `weights.zig` converts a bf16 head to f16
   at load (exact: bf16's 7 mantissa bits fit f16's 10). No shim needed.

`serving_1017.py` writes what the Python cache writer needs and what the server needs besides the weights:

```sh
python3 tools/dsv41-zig/serving_1017.py MODEL_DIR            # the rename shim + chat_template.jinja
python3 tools/dsv41-zig/serving_1017.py MODEL_DIR --check    # exit 0 when both are there and right
```

- `serving-1017.safetensors` (only when the pack has the converter's markov names): byte copies of
  `markov_w1` / `markov_w2` under `embed.weight` / `head.weight`. The loaders map every `*.safetensors` file in the
  folder in name order, a later file winning, and this file sorts after `model-*`, `layer-*` and `native.safetensors`.
- `chat_template.jinja`: `serving/chat_template.jinja`, sha256 `b581d19d...` (ours; checked against DeepSeek's
  reference renderings in the TP4 line, 71/71). The 1017 checkpoint ships no chat template and the server needs one.

Images: the 1017 image routing bias is `kit/vision/gate_bias_vl.safetensors`, sha256 `4745da4b...`, made from the
original 1017 checkpoint (not the EXL3 quant) with `kit/make_bias_vl.py`; point the kit at it (`TF_DS_KIT`) or pass
`--vision-bias`.

The Engram tables are not in the EXL3 packs: pass the original 1017 checkpoint's shards 47 and 48 as `--engram`
(the same as the September packs).

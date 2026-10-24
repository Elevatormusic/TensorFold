# The TFE draft head (DeepSeek-V4.1, Zig engine)

The TFE head is an EAGLE-style draft head. It is trained on the target's hidden states (the EAGLE study's
`eagle_train.py` Head; the ideas come from the public EAGLE papers, not their code). The engine code is in
`zig/src/families/dsv41/eagle.zig` (the pass), `eagle_file.zig` (the head file and its load) and `eagle_check.zig`
(the self-test). The kernels are in `zig/kernels/cuda/dsv41/eagle.cu`.

## Switches

| Variable | Effect |
|---|---|
| `TF_DS_DRAFTER=eagle` | The TFE head drafts and no rank loads DSpark. Unset or `dspark`: DSpark. All ranks must use the same value. |
| `TF_DS_EAGLE_HEAD=FILE` | The head file (`eagle_convert.py` output). |
| `TF_DS_EAGLE_FP8` | Unset: FP8 linears when the file has the FP8 forms. `1`: FP8 (the forms must be there). `0`: bf16. |
| `TF_DS_EAGLE_GEMV` | Linears on the own GEMV: `wo` (default: wo and w2), `all`, or `cublas`. |
| `TF_DS_EAGLE_GRAPH=1` | The steps of a pass as one CUDA graph for each (streams, steps). |
| `TF_DS_EAGLE_TOPK=K` | two_stage heads: K candidates (at most 512) in place of the file's `cand_k`. |
| `TF_DS_EAGLE_CONF=a,b` | The confidence `conf = a * log(p / (1 - p)) + b`. |
| `TF_DS_EAGLE_CHECK=FILE` | Before serving: the head against `eagle_ref.py vectors` output. |
| `TF_DS_EAGLE_TAPDUMP=FILE` | Every absorb writes its tap rows (see "Tap dump"). |

## The head

The head runs on rank 0 only. Rank 0 sends every draft to rank 1 in the verify frame, so rank 1 runs no head and a
pass has no collective.

For a row at position p, with the target's taps at p (the tap layers side by side, the final state last) and the next
token x_{p+1}:

1. g = RMSNorm(fc(taps)).
2. x = w_in [RMSNorm(embed(x_{p+1})) ; RMSNorm(g)].
3. One decoder layer: heads x 128 queries, one key and value head. With qk_norm, an RMSNorm on each query and key
   head. RoPE on 64 of the 128 dims. Attention over the stream's newest window + 1 positions. Then a SwiGLU MLP.
4. logits = up(down(RMSNorm(o))).

A two_stage head takes the top K of those logits (K = `cand_k`, or `TF_DS_EAGLE_TOPK`). It drafts the argmax of their
exact logits from the target's own head (a dense bf16 copy in the head file, on rank 0).

## The chain

A stream's chain starts from its pending token at q0:

- The real rows below q0 have their keys and values in the stream's ring.
- Draft 1 comes from the row q0 - 1.
- Draft j > 1 comes from a row at position q0 + j - 2, with g = the step before's o and the token draft j - 1. Its key
  and value also go into the ring. The next pass's real rows write over them.

`eagle_ref.py` is this order of work in torch (the self-test's reference). It is checked against the training code's
chain (`eagle_ref.py cross`).

The g rows come from each round (absorbRound: the drafting streams' window rows) and from each prompt chunk that
reaches the head (its last ring rows). A pass first runs the rows from the stream's last extended position to q0 - 1
through the layer's input part (their keys and values). Then it runs `steps` single-row steps for all streams at once.

A prompt chunk drops its rows below the replay floor (the prompt length - the model window) at the middle layer. A tap
of a layer below layers / 2 keeps the chunk's rows from its first row, so the absorb reads it at the offset
`ch.start - start0`.

## Linears and FP8

The linears are bf16 cuBLAS calls or, for up to 16 rows, the own GEMV of `eagle.cu`. Both sum in fp32.

With FP8, each linear (fc, w_in, wq, wkv, wo, w13, w2, down, up) is e4m3 with an fp32 scale for each output row (the
file's `NAME.fp8` and `NAME.scale`, made by `eagle_convert.py`). FP8 linears always use the own GEMV. The norms, the
target head's copy and the logits do not change.

The FP8 rule: s = amax(|W| of the row) / 448 in fp32 (1 if the row is 0); q = e4m3 round-to-nearest-even of
fp32(W) / s; the weight is fp32(q) * s.

## Kernels (eagle.cu)

- RMSNorm: a row a block (512 threads).
- RoPE: in place on the first rd features of each head of row r at position pos[r]. The halves are rotated with the
  frequency theta^(-i / (rd / 2)). A row a block.
- Attention: one row and one query head a block (grid rows x heads, 128 threads; head_dim 128). It reads the keys and
  values of the row's stream ring (K, V bf16 [ring rows, kv_heads * 128], the stream's rows from base[r]) at the
  positions max(lo[r], pos[r] - window) .. pos[r] (ring row base + position % ring). The scores are fp32 times
  `scale`, then softmax, then the fp32 sum of the values, bf16 out [rows, heads * 128]. A thread computes the score of
  one key. For the values, each warp sums a quarter of the keys (a lane 4 features), then the 4 warp sums are added.
- top1: a row a block (1024 threads). The largest logit of [n] fp32 (a tie goes to the lower token id), its softmax
  probability p over the n, and conf = a * log(p / (1 - p)) + b (the lane code reads sigmoid(conf) as the draft's
  survival). map: null (logit i is token i), or the row's token ids int64 [n] (a two_stage head's candidates).
- top1_part and top1_join: the top-1 of long rows (the low-rank head's 129280 logits). The part kernel (grid rows x
  parts, 256 threads) finds each part's largest logit (a tie: the lower id) and its sum of exp(logit - that max). The
  join kernel (grid rows, 32 threads) joins the parts and writes the same outputs as top1. Part buffers: pm, pz fp32
  and pi int32 [rows, parts].
- gemv_bf16 and gemv_fp8: y[r, o] = scale[o] * sum_i x[r, i] * W[o, i] for up to 16 rows (fp32 sums). A group of G
  lanes computes one output (G = 32, or the number of 16-byte chunks in a row when that is less). A lane reads every
  G-th chunk. Mode 0: out bf16 = y. Mode 1: out fp32 = y. Mode 2: out bf16 = out + y with one rounding (as cuBLAS beta
  1). x is bf16 [rows, k] with 16-byte aligned rows. 256 threads.
- topk: a two_stage head's candidates, a row a block (1024 threads). The k largest of [n] fp32 logits; a tie at the
  k-th value goes to the lower token ids. The output is int64 ids in cand [rows, k] (the order in a row is not fixed).
  A radix select on the keys (4 passes of 8 bits), then the keys above the k-th and the first of the equal ones in id
  order.
- rescore: a two_stage head's exact logits, out[r, j] = W[cand[r, j]] . x[r] (W = the target head bf16 [vocab, d],
  fp32 sums), fp32 out [rows, k]. Grid rows x ceil(k / 8), 256 threads: a warp a candidate.

## Self-test (TF_DS_EAGLE_CHECK)

The self-test uses slot 0 before serving and resets the slot after. For each case it absorbs the rows, runs one pass,
and compares each step's o, draft and top-1 probability with the vector file. It writes one JSON line for each case
and a verdict on stderr. Pass: every step-1 o is within 3% (max abs difference over max abs), and every draft is equal
up to the first step whose reference top-2 logit gap is less than 0.05 (there, a rounding can select either token).

## Tap dump (TF_DS_EAGLE_TAPDUMP)

Each absorb writes one record: a u32 header length, the JSON header `{kind, slot, start, n, cut, taps, dim}`, then the
rows' taps as bf16 [n, taps * dim] in the engine's tap order, before tap_norm. `kind` is `prompt` or `round`. `cut` is
the number of rows that a prompt chunk dropped at its replay floor. `tfe_tapcheck.py` reads these files.

## Limit: head windows past the model window

A prompt gives the head only its last `sliding_window` rows (128 for V4.1): the rows below the replay floor do not
run the layers that the taps come from. A head with a longer window (for example 511) starts each prompt with 128 keys
and gets its full window only after more tokens. The engine logs a warning at load for such a head. Heads trained with
`--serve-floor 128` see the same history in training.

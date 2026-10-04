# DeepSeek-V4.1-Flash PDMux rollout

The implementation from [#41562](https://github.com/sgl-project/sglang/pull/41562)
is split into four feature commits on current upstream APIs. This document tracks
the review and GPU validation of those layers. The tracking document has its own
branch; it adds no commit to the implementation stack.

## Review stack

| Layer | PR | Review state | Branch | Feature commit |
| --- | --- | --- | --- | --- |
| Core/model: bounded layerwise prefill | [#42507](https://github.com/sgl-project/sglang/pull/42507) | Open, ready | `feat/dsv41-flash-pdmux-core` | `1988d2394c` |
| HiCache progress and SWA ownership | [#42509](https://github.com/sgl-project/sglang/pull/42509) | Open, draft | `fix/dsv41-flash-pdmux-hicache` | `e81176b014` |
| Attention DP and rank alignment | [#42513](https://github.com/sgl-project/sglang/pull/42513) | Open, draft | `feat/dsv41-flash-pdmux-dp` | `cd5788ffd0` |
| Single-layer MTP/EAGLE and DSpark | [#42514](https://github.com/sgl-project/sglang/pull/42514) | Open, draft | `feat/dsv41-flash-pdmux-spec` | `d252b42bb5` |

Every child targets `sgl-project/sglang:main`. Branches contain cumulative prefixes
of the stack until predecessors merge. Each body lists its dependencies and the
focused one-commit diff. Rebase downstream branches onto main and drop merged
predecessors before the next merge. Common runtime changes overlap the GLM stack
[#42411](https://github.com/sgl-project/sglang/pull/42411),
[#42412](https://github.com/sgl-project/sglang/pull/42412),
[#42413](https://github.com/sgl-project/sglang/pull/42413) and
[#42414](https://github.com/sgl-project/sglang/pull/42414); deduplicate after either
stack merges. DeepSeek does not include GLM model or Mamba allocator changes.

## Integration branch

Use `Li-brua:feat/dsv41-pdmux` to validate all four layers together.

- Base: `35f3c96ff4794a4de15daf12caad371084a037ee`.
- Integration HEAD: `d252b42bb579b3986c391f5721b745140901bbc5`.
- Exactly four commits above the base; one feature commit corresponds to each
  child PR. The child refs point to those four successive prefixes.
- Original remote HEAD `13caf792cc30b24b0518a3b9b221ccc88924f785` and divergent
  local development refs were archived before the guarded history rewrite.
  #41562 follows this integration branch and remains a draft reference to the
  split. Merge the child PRs through their review sequence.

```bash
git fetch https://github.com/Li-brua/sglang.git feat/dsv41-pdmux
git switch --detach FETCH_HEAD
git rev-list --count 35f3c96ff4794a4de15daf12caad371084a037ee..HEAD
# Expected: 4
git log --reverse --oneline 35f3c96ff4794a4de15daf12caad371084a037ee..HEAD
```

## Behavior and capability boundaries

PDMux directly uses layerwise prefill. The standard prefill lane, mode parameter,
mode-dependent backend plumbing and their tests/runbook are absent. Token
chunking, layer caps and SM overlap remain supported. With decode present, the
slice budget uses the sum of global prefill tokens; with no decode, ranks execute
the remaining layers together. Compressor planner admission retains its token
cap because layer slicing does not reduce token count.

The core/cache prefixes admit plain TP. The DP layer adds TP/attention DP,
including TP8/DP8 and TP8/DP2 with independent attention-TP groups. The spec layer
adds checkpoint single-NextN MTP/EAGLE and an independent DSpark draft. Eagle
drafts run eager; target and DSpark retain their per-stream graph paths. Final
draft handoff or KV injection is fenced in both stream directions. Intermediate
target slices keep overlapping.

Non-speculative and DSpark target prefills prepare attention-DP inputs once per
token chunk, keep padding across layer slices and restore batch-owned DP sizes,
real device counts and the extend flag after intervening decode. Only the final
slice unpads, before sampling or DSpark draft KV injection. New chunks prepare
fresh batches. Draft, target verify, EAGLE/MTP, LoRA and HiSparse forwards retain
per-call preparation. This update is folded into the DP and spec feature commits;
the integration branch still has four commits above its base.

Multi-layer EAGLE, EAGLE3, adaptive speculative parameters and other speculative
algorithms remain rejected. Existing decoder bounded replay/tail restrictions on
MTP FULL capture and DP still apply. DSpark attention DP also requires the
ordinary adapter's DP LM head configuration; a DeepSeek MoE draft retains its
existing attention-TP-size restriction. EP, CP and DCP are outside this matrix.

## Validation

The final integration tree passed **329 CPU tests and 247 subtests**, with **2
skips**. The core prefix independently passed **109 tests and 40 subtests** after
the architecture accessor exemption was folded into its feature commit. Normal
runtime imports and CPU PyTorch execute the model/state/cache paths; CUDA
streams, kernels and collectives are mocked.

Coverage includes ordinary-forward/split mHC and auxiliary parity, Engram/tail
and interleaved batch state, planner admission, chunk/abort/activation lifetime,
HiCache event cadence and FULL/SWA ownership, DP group/padding/IDLE alignment,
final-only MTP/DSpark handoff and ordinary speculative coordination/backend/graph
regressions. The reuse matrix covers active and IDLE ranks under SUM_LEN and
MAX_LEN padding in TP8/DP8 and TP8/DP2, input tensor identity across slices,
intervening decode metadata and real-row DSpark injection. Python AST, isort, Ruff lint/format, clang-format, codespell, registered
CI checks and `git diff --check` passed. Moving the accessor exemption into the
core commit preserved the final tree exactly.

GPU accuracy, acceptance, concurrent graph replay and performance have not been
verified on the CPU development machine. Upstream full CI requires the repository
`run-ci` label; the initial core run was gated by that missing label. Follow the
[GPU runbook](https://github.com/Li-brua/sglang/blob/d252b42bb579b3986c391f5721b745140901bbc5/test/manual/pdmux/layerwise_prefill_runbook.md)
on the integration branch and compare each algorithm against its ordinary
scheduler with the same checkpoint and workload.

- [ ] Plain TP1/TP8 greedy output and concurrent long prefill/decode.
- [ ] HiCache reload, eviction and repeated FULL/SWA ownership; include
  write-back/storage/buffer boundaries beyond the write-through manual cases.
- [ ] TP8/DP8 and TP8/DP2, uneven and peer-only work, all-IDLE collective liveness.
- [ ] Single-NextN MTP and DSpark output/acceptance, final-only extension/injection
  and backend/graph selection.
- [ ] Token chunks, parked chunks, aborts and HiCache on/off in both SM layouts.
- [ ] Prepare/copy/unpad calls per token chunk under sustained decode, for plain
  and DSpark attention DP; compare greedy output and DSpark acceptance.
- [ ] TTFT, ITL p50/p99, output tokens/s and per-rank peak memory against matching
  ordinary-scheduler baselines.

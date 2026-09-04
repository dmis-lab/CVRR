# Analysis

The analysis package is grouped by purpose:

```text
core/          checkpoint runtime, datasets, statistics, shard merge
causal/        latent interventions and residual swaps
recurrence/    depth, rereading, and component controls
diagnostics/   learned-transition and causal visual-region measurements
localization.py
efficiency.py
```

All model-facing commands save aggregate statistics and raw per-example data.
V* causal analyses use the restricted first-option-token readout; those scores
must not be mixed with generation-based benchmark scores.

Run commands from the repository root:

```bash
scripts/analyze.sh --help
```

## Boundary localization

```bash
scripts/analyze.sh localize \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --pairs /path/to/localization_pairs.json \
  --image-root /path/to/extracted_viscot \
  --out results/localization.json
```

The pair file must contain exact same-question examples with different images
and answers. The command performs bidirectional image-row activation patching
one layer at a time.

## Causal analyses

```bash
scripts/analyze.sh causal \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --out results/causal.json

scripts/analyze.sh residual-swap \
  --checkpoint /path/to/checkpoint \
  --benchmark mmvp \
  --mmvp-root /path/to/MMVP \
  --out results/residual_swap.json
```

The causal command evaluates clean, matched-swap, blank, and norm-matched-noise
states under strict and restored-prefix answer paths. Residual swap preserves
the exact same-question text anchor and exchanges only the operational
image-conditioned residual.

## Recurrence analyses

```bash
scripts/analyze.sh recurrence \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --pairs /path/to/localization_pairs.json \
  --image-root /path/to/extracted_viscot \
  --out results/recurrence.json

scripts/analyze.sh reread \
  --checkpoint /path/to/checkpoint \
  --pairs /path/to/localization_pairs.json \
  --image-root /path/to/extracted_viscot \
  --out results/reread.json

scripts/analyze.sh persistent \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --out results/visual_access.json

scripts/analyze.sh components \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --out results/components.json
```

These commands measure recurrence depth, crossed-state visual correction,
compute-matched visual-access controls, and matched structural ablations.

## Internal diagnostics and efficiency

```bash
scripts/analyze.sh transition-activation \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --out results/transition.json

scripts/analyze.sh query-channel \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --out results/query_channels.json

scripts/analyze.sh visual-map \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --indices 6 115 \
  --out results/visual_regions.json

scripts/analyze.sh efficiency \
  --checkpoint /path/to/checkpoint \
  --vstar-root /path/to/vstar_bench \
  --steps 1 2 3 4 6 8 \
  --out results/efficiency.json
```

Transition and query-channel values are functional adapter corrections, not
attention importance. Visual-region values are output changes caused by
blocking selected question-to-image edges.

Most commands support independent shards. Give each shard a unique path and
merge compatible outputs with:

```bash
scripts/analyze.sh merge \
  --inputs results/run.shard*.json \
  --out results/run.merged.json
```

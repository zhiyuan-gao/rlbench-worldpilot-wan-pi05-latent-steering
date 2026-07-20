# FCRF-v1 on `hpc-4xh100-nvl`

## Scope

This is an additive implementation. No existing Block12, early-prefix,
OpenVLA-OFT, data, or evaluation file is changed.

New files:

```text
src/rlbench_worldpilot_wan_pi05/fcrf.py
src/rlbench_worldpilot_wan_pi05/modeling_fcrf.py
src/rlbench_worldpilot_wan_pi05/train_fcrf_v1.py
scripts/train_fcrf_v1_hpc_4xh100.sh
scripts/smoke_fcrf_v1.sh
scripts/eval_fcrf_v1_flow.sh
tests/test_fcrf.py
tests/smoke_fcrf_ddp.py
docs/FCRF_V1_HPC_4XH100.md
CODEX_HANDOFF_FCRF_V1_HPC_4XH100.md
```

## Fixed method

```text
v_base       = frozen RLBench-finetuned pi0.5 flow
delta_v      = FCRF(suffix_out_base, 18 WAN view/time tokens, flow time)
gate         = per-sample sigmoid gate
v_final      = v_base + gate * delta_v
loss         = flow_matching_mse + 1e-4 * correction_l2
```

Only these modules are trainable:

```text
fresh WAN token encoder
flow-time MLP
flow residual cross-attention/MLP/head
sample-conditioned gate
```

The full pi0.5-ft model, action expert, action projections, and WAN video model
are frozen. The residual output projection is zero initialized, so
`v_final == v_base` exactly before training. The gate bias is `-2.2`
(`sigmoid ~= 0.1`). Current GFPI token-encoder weights are not reused.

## Required checkpoint

Set a separate variable so that pi0.5 base cannot be selected accidentally:

```bash
export PI05_FT_PYTORCH_WEIGHT_PATH=/path/to/converted_rlbench_pi05_ft_checkpoint
test -f "${PI05_FT_PYTORCH_WEIGHT_PATH}/model.safetensors"
```

## Smoke

After sourcing the existing HPC path file:

```bash
source scripts/hpc_paths.sh
bash scripts/smoke_fcrf_v1.sh
```

The smoke must report:

```text
initial_max_abs_flow_difference = 0
vanilla_off_max_abs_loss_difference = 0
base_grad_tensors = 0
fcrf_nonzero_grad_tensors > 0
all trainable parameter names start with fcrf.
```

## 2k pilot on 4x H100

```bash
source scripts/hpc_paths.sh
export PI05_FT_PYTORCH_WEIGHT_PATH=/path/to/converted_rlbench_pi05_ft_checkpoint
export EXP_NAME=selected10_fcrf_v1_pilot2k

NPROC_PER_NODE=4 bash scripts/train_fcrf_v1_hpc_4xh100.sh
```

Defaults:

```text
global batch        128
optimizer steps     2000
sample exposures    256000
warmup              200
checkpoints         500, 1000, 1500, 2000
residual penalty    1e-4
```

Do not reuse the old 10000-step warmup for this pilot.

## Offline diagnostics

Run the same 400 validation rows at each decision checkpoint:

```bash
for step in 500 1000 2000; do
  EVAL_CHECKPOINT="${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/${step}" \
  bash scripts/eval_fcrf_v1_flow.sh
done
```

Each output compares the same frozen base flow against:

```text
off
matched WAN latent
same-task, preferably same-event-index, different-episode WAN latent
```

Use the `physical_*` metrics (the first seven dimensions executed by RLBench)
as the primary decision signal. Unprefixed metrics average all 32 π0.5 model
dimensions, including 25 padding dimensions.

Continue directly to 10k only if the trend is stable:

```text
physical_matched_mse < physical_off_mse
physical_matched_mse < physical_shuffled_mse
physical_correction_cosine > 0
residual/correction norms remain finite and controlled
gate does not collapse to all-zero or all-one
```

If matched improves over off and cosine is positive, but matched versus
shuffled is still inconclusive while the gap grows from 500 to 2000, extend
only to 5k. If matched remains equal to shuffled at all three checkpoints,
stop and inspect future/action alignment before spending the 10k budget.

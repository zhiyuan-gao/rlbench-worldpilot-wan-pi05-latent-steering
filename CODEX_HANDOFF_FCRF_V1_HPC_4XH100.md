# Codex Handoff: FCRF-v1 on `hpc-4xh100-nvl`

Date: 2026-07-20

This file is the operational source of truth for the next FCRF-v1 experiment
on a single 4x H100 NVL 94GB node.

## 1. Scope and non-negotiable constraints

Work on exactly this repository and branch:

```text
repository  zhiyuan-gao/rlbench-worldpilot-wan-pi05-latent-steering
branch      hpc-4xh100-nvl
hardware    1 node, 4x H100 NVL 94GB
```

Before doing anything:

```bash
git checkout hpc-4xh100-nvl
git pull --ff-only
git status --short
git log -1 --oneline
```

The worktree must be clean. Do not switch to `main`, do not merge the whole
`main` branch, and do not change the existing early-prefix, Block12,
OpenVLA-OFT, WAN export, or old online-eval implementation.

Ignore all OpenVLA-OFT files for this experiment. FCRF-v1 is a π0.5-ft
experiment.

Do not run the old training entry:

```text
scripts/train_worldpilot_wan_pi05_torch.sh
```

The FCRF entry is:

```text
scripts/train_fcrf_v1_hpc_4xh100.sh
```

## 2. Why this experiment exists

The selected10 results show task-level complementarity.

```text
π0.5-ft total                   142/250 = 56.8%
WAN hidden-token action policy  99/250 = 39.6%
task-wise oracle choice         170/250 = 68.0%
```

The WAN action policy is especially strong on:

```text
sweep_to_dustpan       23/25 = 92%
slide_block_to_target  22/25 = 88%
stack_wine              8/25 = 32%
```

π0.5-ft is much stronger on tasks such as `put_money_in_safe`,
`reach_and_drag`, `push_buttons`, and `open_drawer`.

Block12/GFPI changed the task distribution but did not improve the total over
π0.5-ft. Its prefix injection and single global scalar gate could help some
WAN-favored tasks while damaging π0.5-favored tasks.

The previous 400-sample GFPI flow diagnostic found:

```text
matched MSE                  0.01191337
same-skill shuffled MSE      0.01191960
matched improvement rate     51.75%
correction cosine mean       +0.19031
P(cosine > 0)                69.75%
```

Interpretation:

- the learned WAN correction was directionally useful on average;
- matched was not meaningfully better than same-skill shuffled;
- therefore the GFPI prefix fuser was not proven to use episode-specific
  future information;
- FCRF uses a fresh token encoder and does not load GFPI fuser weights.

FCRF-v1 is a cautious test of whether WAN information can make a small,
sample-conditioned action-flow correction without overwriting π0.5-ft.

## 3. Exact FCRF-v1 method

The frozen base computes:

```text
v_base = pi0.5-ft(observation, language, noisy_action x_t, flow_time t)
```

The new branch computes:

```text
delta_v = residual_branch(suffix_out_base, WAN_tokens, t)
gate    = sigmoid(gate_branch(suffix_out_base, WAN_tokens, t))
v_final = v_base + gate * delta_v
```

The loss is:

```text
flow_mse = mean((v_final - target_flow)^2)
loss     = flow_mse + 1e-4 * mean((gate * delta_v)^2)
```

`target_flow` is the unchanged OpenPI flow-matching target:

```text
target_flow = noise - action
```

The OpenPI config currently has:

```text
action_horizon = 1
model action_dim = 32
physical RLBench action dimensions = first 7
```

The FCRF gate shape is `[B,1,1]`. It is one gate per sample and flow timestep,
broadcast across the single action token and 32 model action dimensions.

### WAN tokens

The aligned cache tensor is:

```text
[B,V,C,T,H,W] = [B,3,16,6,32,32]
```

The fresh encoder keeps view and latent time separate:

```text
3 views x 6 latent times = 18 tokens
```

For each `(view,time)` pair, `[C,H,W]` is flattened and projected to the action
expert hidden width. Learned view and time embeddings are added before
cross-attention.

The six WAN time positions span the generated future toward the current event
end. The supervised action remains only the next waypoint. FCRF can attend to
the earlier WAN time tokens, but this is not a proof that event-end future is
properly aligned with the early waypoint. Matched-versus-shuffled diagnostics
must establish that.

### Initialization and base protection

The residual output projection is initialized to exactly zero. The gate output
weight is initialized to zero and its bias is `-2.2`, giving an initial gate of
about `0.1`.

Therefore, before the first optimizer step:

```text
delta_v = 0
v_final = v_base exactly
```

The following are frozen:

```text
π0.5-ft PaliGemma vision-language model
π0.5-ft action expert
action_in_proj
action_out_proj
π0.5 flow-time MLP
WAN video model
```

Only parameters named `fcrf.*` are trainable:

```text
fresh WAN token encoder
FCRF flow-time MLP
residual cross-attention
residual MLP and output head
sample-conditioned gate
```

The current real-checkpoint smoke reports approximately:

```text
frozen parameters       3,616,757,520
trainable parameters       33,688,609
```

At the zero-initialized first backward pass, only the residual output weight
and bias have non-zero gradients. This is expected. Once they become non-zero,
gradients reach the upstream FCRF modules and gate.

## 4. FCRF files

```text
src/rlbench_worldpilot_wan_pi05/fcrf.py
    Pure PyTorch 18-token encoder, residual branch, and gate.

src/rlbench_worldpilot_wan_pi05/modeling_fcrf.py
    PI0FCRFV1Pytorch. Extracts the frozen base suffix hidden/flow, applies
    FCRF, and implements FCRF action denoising.

src/rlbench_worldpilot_wan_pi05/train_fcrf_v1.py
    FCRF-only optimizer, frozen-scope checks, DDP training, smoke, checkpoint
    resume, and matched/off/same-skill-shuffled diagnostics.

scripts/train_fcrf_v1_hpc_4xh100.sh
    Main 4x H100 launcher. Defaults to the 2k pilot.

scripts/smoke_fcrf_v1.sh
    One-GPU real-model forward/backward and equivalence smoke.

scripts/eval_fcrf_v1_flow.sh
    One-GPU 400-sample offline diagnostic launcher.

tests/test_fcrf.py
    Small tensor-level unit tests.

tests/smoke_fcrf_ddp.py
    CUDA DDP forward/backward/optimizer synchronization smoke.

docs/FCRF_V1_HPC_4XH100.md
    Short command reference.

CODEX_HANDOFF_FCRF_V1_HPC_4XH100.md
    This operational handoff and decision protocol.
```

## 5. Required HPC inputs

Machine-specific paths are intentionally not committed. If this is a fresh
checkout, create the local path file from the tracked template, edit it for
the node, and keep it untracked:

```bash
test -f scripts/hpc_paths.sh || cp scripts/hpc_paths_example.sh scripts/hpc_paths.sh
${EDITOR:-vi} scripts/hpc_paths.sh
```

Then source it:

```bash
source scripts/hpc_paths.sh
```

Then define the FCRF-specific base checkpoint explicitly:

```bash
export PI05_FT_PYTORCH_WEIGHT_PATH=/path/to/19Julipi05_init_39999
```

This must be the converted RLBench-finetuned π0.5 checkpoint, not π0.5 base.
The expected file is:

```bash
test -f "${PI05_FT_PYTORCH_WEIGHT_PATH}/model.safetensors"
```

The local source checkpoint used during implementation was:

```text
/raid/home/than/zhiyuan/worldpilot_files/19Julipi05_init_39999
```

Do not assume that local path exists on the HPC. Upload it or point the
variable to the HPC copy.

Also verify:

```bash
test -f "${MANIFEST_PATH}"
test -f "${EVENT_MANIFEST_PATH}"
test -f "${WAN_LATENT_CACHE_ROOT}/sample_index_train.jsonl"
test -f "${WAN_LATENT_CACHE_ROOT}/sample_index_val.jsonl"
test -f "${HF_LEROBOT_HOME}/${LEROBOT_REPO_ID}/meta/info.json"
test -f "${ASSETS_BASE_DIR}/pi05_rlbench_waypoint_h1/${LEROBOT_REPO_ID}/norm_stats.json"
```

The train and val WAN caches must be fully exported with:

```text
goal mode             event_end
backend               wan-diffusers
WAN denoise steps     1
latent shape          3,16,6,32,32
```

Do not train with `--allow-missing-latents` or dummy latents.

OpenPI normalization statistics are a separate input; the local
`19Julipi05_init_39999` directory contains model weights but not this file. The
local stats used for the smoke came from:

```text
/raid/home/than/zhiyuan/worldpilot_files/gfpi_frozen_block12_step20000/assets/rlbench/selected10_pi05_waypoint_h1/norm_stats.json
```

On the HPC, put the file at the exact `ASSETS_BASE_DIR` path checked above. If
no trusted copy is available, run the π0.5 baseline repository's
`compute_norm_stats.sh` with the same `pi05_rlbench_waypoint_h1` config and
LeRobot repo, then place its output at that path. Do not use
`--skip-norm-stats` for smoke, training, or offline evaluation.

## 6. Preflight and smoke

Run unit tests without loading the 3.6B model:

```bash
source scripts/hpc_paths.sh
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/src:${OPENPI_DIR}/src" \
uv run --directory "${OPENPI_DIR}" python -m pytest -q \
  "${REPO_ROOT}/tests/test_fcrf.py"
```

Then run the real π0.5-ft/WAN-cache smoke on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_fcrf_v1.sh
```

Required smoke results:

```text
smoke                                      passed
vanilla_off_max_abs_loss_difference        0.0
initial_max_abs_flow_difference             0.0
base_grad_tensors                           0
fcrf_nonzero_grad_tensors                   > 0
all trainable names                         start with fcrf.
```

Do not start the pilot if any equivalence or frozen-scope check fails.

Run a lightweight FCRF-only DDP smoke before loading the full model on four
GPUs:

```bash
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/src" \
uv run --directory "${OPENPI_DIR}" torchrun \
  --standalone --nnodes=1 --nproc_per_node=4 \
  "${REPO_ROOT}/tests/smoke_fcrf_ddp.py"
```

It must report `smoke=passed` and `world_size=4`.

## 7. Run the 2k pilot

Use a new experiment name and confirm the output directory does not contain an
unrelated run:

```bash
export EXP_NAME=selected10_fcrf_v1_pilot2k
export WANDB_DIR=/path/to/hpc/wandb

NPROC_PER_NODE=4 bash scripts/train_fcrf_v1_hpc_4xh100.sh
```

Fixed pilot defaults:

```text
GPUs                    4
global batch            128
local batch             32 per H100
optimizer steps         2000
sample exposures        256000
warmup steps            200
peak LR                 5e-5 from the OpenPI config
residual penalty        1e-4
checkpoint steps        500,1000,1500,2000
precision               bfloat16
```

The old 10000-step warmup is not valid for a 2k pilot.

Expected checkpoint layout:

```text
${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/500
${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/1000
${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/1500
${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/2000
```

The training log should include:

```text
train/flow_mse
train/base_mse
train/correction_cosine
train/gate_mean
train/correction_norm
train/residual_penalty
train/grad_norm
```

The console/W&B training metrics are rank-0 minibatch diagnostics. The fixed
validation diagnostic below is the decision source.

Stop immediately on NaN/Inf, cache mismatch, non-FCRF trainable parameters, or
unbounded correction/residual norms.

## 8. Run fixed offline diagnostics

Evaluate exactly the same val rows at steps 500, 1000, and 2000:

```bash
for step in 500 1000 2000; do
  EVAL_CHECKPOINT="${CHECKPOINT_BASE_DIR}/pi05_rlbench_waypoint_h1/${EXP_NAME}/${step}" \
  MAX_EVAL_SAMPLES=400 \
  SHUFFLE_SEED=0 \
  bash scripts/eval_fcrf_v1_flow.sh
done
```

The diagnostic runs on one GPU and computes the frozen base only once per
batch. The same base flow, target, action noise, and flow time are used for:

```text
off       v_base
matched   v_base + FCRF(sample's WAN latent)
shuffled  v_base + FCRF(other episode's WAN latent)
```

Shuffling never crosses task. It uses a different episode and preserves the
same event index whenever a valid derangement exists. The JSON reports how
many rows needed task-level fallback.

The evaluation loader does not drop the final partial batch. With the current
400-row validation set and batch 32, it evaluates `12x32 + 16 = 400`, not 384.
Check:

```text
evaluated_samples = 400
overall.num_samples = 400
```

Important output fields:

```text
off_mse
matched_mse
shuffled_mse
matched_improvement_over_off
matched_improvement_over_shuffled
correction_cosine
positive_cosine
gate
correction_norm
residual_norm
```

Every MSE, cosine, positive-cosine rate, correction norm, norm ratio, and
improvement field is also emitted with these prefixes:

```text
physical_   first 7 executed RLBench dimensions
xyz_        dimensions 0:3
rotvec_     dimensions 3:6
gripper_    dimension 6
```

The unprefixed metrics cover all 32 π0.5 model dimensions. The extra 25
dimensions are padding and are not executed by RLBench. Use `physical_*` as
the primary offline decision signal; use the all-32 metrics only as a model
training diagnostic. Inspect both overall and per-task values.

## 9. Decision rule after 2k

Continue directly to 10k only if the 500 -> 1000 -> 2000 trend is stable and:

```text
physical_matched_mse < physical_off_mse
physical_matched_mse < physical_shuffled_mse
physical_correction_cosine > 0
correction/residual norms are controlled
gate is finite and not collapsed to all-zero or all-one
```

Also check the task pattern. A useful FCRF should preserve π0.5-favored tasks
while improving WAN-favored tasks. For the later online pilot, prioritize:

```text
WAN-favored:   sweep_to_dustpan, slide_block_to_target
π0.5-favored:  put_money_in_safe, reach_and_drag
```

If matched improves over off and cosine is positive, but matched-versus-
shuffled is still inconclusive while the gap is growing, extend only to 5k.

If matched remains equal to shuffled at all checkpoints, stop. Do not continue
just because train loss falls. First inspect event-end/next-waypoint alignment,
sample/cache indexing, and WAN time-token use.

If shuffled is better than matched, stop and treat this as evidence of an
alignment or conditioning problem.

## 10. Continue the same run to 10k

Do not restart from scratch after a positive 2k pilot. Preserve the optimizer
state and continue the same experiment:

```bash
export EXP_NAME=selected10_fcrf_v1_pilot2k

NPROC_PER_NODE=4 \
bash scripts/train_fcrf_v1_hpc_4xh100.sh \
  --resume \
  --num-train-steps 10000 \
  --save-interval 2000 \
  --keep-period 2000
```

The scheduler is step-derived; it keeps the 200-step warmup and does not start
a new warmup on resume.

Do not automatically continue to 20k. First run the 10k offline diagnostic and
the frozen formal online protocol.

## 11. What is not yet provided by this branch

This branch contains FCRF training, checkpoint resume, offline diagnostics,
and `PI0FCRFV1Pytorch.sample_actions()`.

It does not yet contain a dedicated FCRF version of the formal RPC per-event-v1
online launcher currently used for the final Block12 evaluation. Do not use
the old per-step online evaluator for a paper comparison without first porting
the frozen formal protocol to the FCRF policy loader.

The 2k offline pilot can proceed without that port. If the 2k signal is
positive, the next code task is a new additive FCRF RPC policy/eval entry; do
not modify the existing evaluator in place.

FCRF-v1 also does not consume the output of the separate WAN diffusion action
policy. It uses WAN future-video hidden information to learn a residual on
π0.5-ft. Therefore, the WAN policy's 92% `sweep_to_dustpan` result is evidence
of representational potential, not a success rate that transfers automatically.

## 12. Audit status before HPC handoff

The implementation has been checked for:

```text
fresh 18-token WAN encoder
zero-initialized exact identity
vanilla π0.5/OpenPI off equivalence
π0.5/action-expert/action-head freezing
FCRF-only optimizer scope
flow-time-conditioned residual and gate
matched/off/shuffled use of identical base flow/noise/time
same-task, different-episode shuffle
no dropped validation remainder
checkpoint resume to arbitrary num_train_steps
2-GPU bf16 DDP forward/backward/update parameter synchronization (local A40)
non-finite loss failure
5D/6D WAN latent handling in sample_actions
```

Local validation used the real converted `19Julipi05_init_39999` checkpoint and
real WAN validation cache. The 4-GPU DDP smoke, unit/static checks, and
real-model smoke must be re-run on the H100 node before launching the 2k job.

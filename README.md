# SoftRHCR

**Soft-RHCR** — A Lifelong-MAPF (Multi-Agent Path Finding) RL research framework for warehouse multi-AGV systems.

## Setup
Make sure LMAPF-Simulator has been installed in your environment
```bash
# conda create -n softrhcr python=3.13
# Install LMAPF-Simulator first
conda activate softrhcr
pip install -e .
```

## Training

```bash
# Train with a JSON config
softrhcr-train SoftRHCR/config/usercfg/TrainConfig.json

# Resume from a checkpoint
softrhcr-train SoftRHCR/config/usercfg/TrainConfig.json --checkpoint /path/to/checkpoint.pth

# Hit Ctrl+C during training to save an interrupt checkpoint automatically.
# Resume later with --checkpoint pointing to it.

# Commonly used CLI overrides:
#   --num-agvs 10 --seed 42 --device cuda
#   --total-train-steps 2000000 --lr 3e-4 --save-interval-steps 100000
#   --algo-override force_rl_prob_start=0.9 --algo-override planner_aux_loss=consistency
```

## Evaluation

```bash
softrhcr-evaluate SoftRHCR/config/usercfg/EvaluateConfig.json \
    --model-path /path/to/model.pth \
    --eval-episodes 100
```

## Runtime Visualization

```bash
softrhcr-experiment SoftRHCR/config/usercfg/ExpConfig.json \
    --model-path /path/to/model.pth --render
```

## Configuration

Config files are JSON, split into `common` (environment) and `specific` (run-mode) sections. Key fields:

| Field | Description |
|-------|-------------|
| `algorithm` | Algorithm: `ippo`, `mappo`, `soft_rhcr`, `soft_rhcr_mappo`, `gateblend`, `gateblend_mappo`, `follow_planner` |
| `num_agvs` | Number of AGVs |
| `map_size` | Map size: `"short"`, `"long"`, `"wide"` |
| `planner_type` | Path planner: `AStar`, `RHCR`, `PBS` etc. |
| `total_train_steps` | Total training steps |
| `checkpoint_path` | Checkpoint path for resuming |
| `device` | Device: `cuda`, `cpu`, `auto` |


## Project Structure

```
SoftRHCR/
├── algorithms/          # RL algorithm implementations
│   ├── IPPO/            #   Independent PPO
│   ├── MAPPO/           #   Centralized PPO
│   ├── SoftRHCR/        #   Soft RHCR (core algorithm)
│   ├── GateBlend/       #   Gated blending baseline
│   ├── FollowPlanner/   #   Pure planner baseline
│   └── PPO/             #   Shared PPO update logic
├── config/              # Configuration system & user configs
├── modules/             # Shared modules (network, reward, logging, etc.)
└── scripts/             # CLI entry points (train, evaluate, experiment)
```

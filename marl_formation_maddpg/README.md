# MARL Formation — MADDPG on PettingZoo MPE `simple_spread_v3`

This repo trains **MADDPG** (centralized critic, decentralized actors) on the cooperative
**Simple Spread** multi-agent environment using **AgileRL** + **PettingZoo**.

- Algorithm: MADDPG (continuous actions)
- Env: PettingZoo MPE `simple_spread_v3` (3 agents, 3 landmarks, continuous actions)
- OS: Windows or macOS/Linux (Windows-friendly)
- Output: learning curves, evaluation metrics, and a short GIF

## 1) Quickstart (Conda on Windows/macOS/Linux)

```bash
# Create env
conda create -y -n marl python=3.11
conda activate marl

# Install PyTorch (CPU) — if you have NVIDIA GPU, install the CUDA build from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install libs
pip install pettingzoo[mpe]>=1.23.1 supersuit>=3.9.0 gymnasium>=0.28.1             agilerl>=2.2.1 numpy>=1.24 matplotlib imageio pillow tqdm pyyaml

# (Optional) For GIFs with higher quality
pip install moviepy
```

> **Note**: `simple_spread_v3` recently moved under `mpe2` in newer PettingZoo versions. 
This repo auto-detects and imports from `mpe2` when available. citeturn0search0

## 2) Train

```bash
python -m src.train --episodes 300 --max-steps 200 --save-dir runs/exp1
```

This will:
- create the Simple Spread parallel env (3 agents, continuous actions),
- train a single MADDPG agent population,
- save checkpoints to `runs/exp1/models/`,
- log metrics to `runs/exp1/metrics.jsonl`,
- save a learning curve PNG to `runs/exp1/curves.png`.

## 3) Evaluate + Make GIF

```bash
# Evaluate the latest checkpoint
python -m src.eval --model runs/exp1/models/MADDPG_latest.pt --episodes 10 --gif results/gifs/episode.gif
```

## 4) Repo Layout

````
marl-formation-maddpg/
├─ README.md
├─ environment.yml                # optional conda spec
├─ src/
│  ├─ train.py                    # training loop (AgileRL + PettingZoo)
│  ├─ eval.py                     # load checkpoint, evaluate, record frames
│  └─ utils/
│     ├─ env_utils.py             # env creation (mpe/mpe2 compatible)
│     └─ plot.py                  # learning curve plotting
├─ scripts/
│  ├─ run_train.bat               # Windows helper
│  └─ run_eval.bat                # Windows helper
└─ results/
   └─ gifs/                       # output GIFs
````

## 5) References

- PettingZoo MPE **Simple Spread** docs (action/observation, continuous controls). citeturn4search2turn4search10  
- AgileRL **MADDPG** tutorial & API (multi-agent, PettingZoo parallel API). citeturn3view0turn4search11

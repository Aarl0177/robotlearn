@echo off
REM Windows helper: evaluate checkpoint and save GIF
conda activate marl
python -m src.eval --model runs/win_exp/models/MADDPG_latest.pt --episodes 5 --gif results/gifs/win_episode.gif

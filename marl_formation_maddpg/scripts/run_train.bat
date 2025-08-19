@echo off
REM Windows helper: train MADDPG on Simple Spread
conda activate marl
python -m src.train --episodes 300 --max-steps 200 --save-dir runs/win_exp

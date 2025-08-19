# src/utils/plot.py
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(jsonl_path: str, out_png: str):
    steps, returns = [], []
    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            if "global_step" in rec and "avg_return" in rec:
                steps.append(rec["global_step"])
                returns.append(rec["avg_return"])
    if not steps:
        print("No records found in", jsonl_path)
        return
    steps = np.array(steps); returns = np.array(returns)
    plt.figure()
    plt.plot(steps, returns, linewidth=2)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Episodic Return")
    plt.title("MADDPG on Simple Spread (avg return)")
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

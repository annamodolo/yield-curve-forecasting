"""
Plotting utilities. Saves plots in results/.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def plot_pred_vs_actual(y_true, y_pred, title: str, out_path: str) -> None:
    """
    Plot predicted vs actual values and save the figure to disk.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

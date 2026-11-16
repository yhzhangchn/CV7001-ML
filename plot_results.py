"""Draw the prediction and residual panels used in the report."""
from pathlib import Path
import math
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np


def save_prediction_figure(measured, predicted, output_stem, model_name, language="en"):
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - measured
    times = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/times.ttf"
    if times.is_file():
        font_manager.fontManager.addfont(str(times))
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "STIXGeneral"
    style = {"font.family": family, "font.size": 9.5, "axes.labelsize": 10.5,
             "axes.titlesize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
             "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
             "mathtext.fontset": "stix"}
    chinese = language == "zh"
    cjk = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc") if chinese else None
    labels = ("实测抗压强度（MPa）", "预测抗压强度（MPa）", "预测误差（MPa）") if chinese else (
        "Measured strength (MPa)", "Predicted strength (MPa)", "Residual (MPa)")
    titles = ("(a) 预测值与实测值", "(b) 预测残差") if chinese else (
        "(a) Predicted vs. measured", "(b) Prediction residuals")
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
        upper = math.ceil((max(measured.max(), predicted.max()) + 3) / 5) * 5
        lower = min(0, math.floor(min(measured.min(), predicted.min()) / 10) * 10)
        limit = max(10, math.ceil(np.abs(error).max() / 10) * 10)
        axes[0].scatter(measured, predicted, s=19, c="#346D90", alpha=.85,
                        edgecolors="white", linewidths=.25, zorder=3)
        axes[0].plot([lower, upper], [lower, upper], color="#666666",
                     linewidth=.9, linestyle=(0, (4, 3)), zorder=2)
        axes[0].set(xlim=(lower, upper), ylim=(lower, upper))
        rmse = float(np.sqrt(np.mean(error**2)))
        r2 = float(1 - np.sum(error**2) / np.sum((measured-measured.mean())**2))
        axes[0].text(.95, .055, f"{model_name}\nR² = {r2:.3f}\nRMSE = {rmse:.2f} MPa",
                     transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8.8,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
        axes[1].scatter(measured, error, s=19, c="#B66A50", alpha=.85,
                        edgecolors="white", linewidths=.25, zorder=3)
        axes[1].axhline(0, color="#666666", linewidth=.9, linestyle=(0, (4, 3)), zorder=2)
        axes[1].set(xlim=(lower, upper), ylim=(-limit, limit))
        axes[1].text(.95, .945, f"n = {len(measured)}", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=8.8)
        for index, ax in enumerate(axes):
            ax.set_xlabel(labels[0], fontproperties=cjk)
            ax.set_ylabel(labels[index+1], fontproperties=cjk)
            ax.set_title(titles[index], fontproperties=cjk, pad=7)
            ax.set_box_aspect(1)
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.tick_params(direction="in", length=3, width=.65, top=True, right=True, pad=3)
            ax.grid(color="#dddddd", linewidth=.35, alpha=.8, zorder=0)
            for spine in ax.spines.values():
                spine.set_color("#333333")
        output_stem = Path(output_stem)
        fig.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=.04)
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04)
        plt.close(fig)

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import DATASETS, OUT_PATH


if not os.path.exists(f"{OUT_PATH}/saturation/plots"):
    os.makedirs(f"{OUT_PATH}/saturation/plots")


def plot_saturation_heatmap(dataset_name):
    n_trees = [t for t in range(10, 110, 10)]

    def data_assembler(model_id):
        dmatrix = None
        agg = False

        for i, n in enumerate(reversed(n_trees)):
            m_log = pd.read_csv(
                f"{OUT_PATH}/saturation/final/mean_{dataset_name}_{model_id}_t{n}.csv"
            )

            if dmatrix is None:
                if m_log["n_samples"].values[-1] > 50000:
                    agg = True
                dmatrix = np.zeros((len(n_trees), len(m_log)))
            dmatrix[i, :] = m_log["RMSE"].values
        if m_log["n_samples"].values[-1] > 50000:
            # Aggregate every 2000 instances
            agg = True
            dmatrix = dmatrix.reshape(-1, dmatrix.shape[1] // 20, 20).mean(axis=2)
        else:
            # Aggregate every 500 instances
            dmatrix = dmatrix.reshape(-1, dmatrix.shape[1] // 5, 5).mean(axis=2)

        dmatrix = pd.DataFrame(
            dmatrix,
            columns=[
                500 * i for i in range(1, dmatrix.shape[1] + 1)
            ] if not agg else [
                2000 * i for i in range(1, dmatrix.shape[1] + 1)
            ]
        )

        # Normalize data
        dmatrix = (dmatrix.div(dmatrix.iloc[-2, :]) - 1) * 100

        return dmatrix

    model_matcher = {
        "ARF-abs": "ARF",
        "XT": "OXT",

        # "ARF-abs-mean": "ARF",
        # "XT-mean": "OXT"
    }
    plot_data = {}

    for model_id in model_matcher:
        plot_data[model_id] = data_assembler(model_id)

    min_ = math.inf
    max_ = -math.inf

    for dmat in plot_data.values():
        mn, mx = dmat.values.min(), dmat.values.max()

        if mn < min_:
            min_ = mn

        if mx > max_:
            max_ = mx

    for model_id, model_name in model_matcher.items():
        dmatrix = plot_data[model_id]

        fig, ax = plt.subplots(figsize=(10, 4), dpi=600)
        ax = sns.heatmap(
            dmatrix, linewidth=0.05, yticklabels=list(reversed(n_trees)),
            xticklabels=10, cbar_kws={
                "label": "Variation in RMSE",
                "format": "%.0f%%"
            },
            ax=ax, center=0.0, vmin=min_, vmax=max_
        )
        for i in range(dmatrix.shape[0] + 1):
            ax.axhline(i, color='white', lw=1)

        ax.yaxis.get_ticklabels()[-2].set_color("darkgreen")
        ax.yaxis.get_ticklabels()[-2].set_weight("bold")

        ax.set_title(f"{DATASETS[dataset_name]} - {model_name}")
        # ax.tick_params(bottom=False)

        plt.yticks(rotation=0)
        plt.savefig(
            f"{OUT_PATH}/saturation/plots/heatmap_{dataset_name}_{model_id}.png",
            bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    plot_saturation_heatmap("cal_housing")
    plot_saturation_heatmap("friedman")
    plot_saturation_heatmap("friedman_gra")

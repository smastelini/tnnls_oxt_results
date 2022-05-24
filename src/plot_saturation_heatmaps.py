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
            # Aggregate every 1000 instances
            agg = True
            dmatrix = dmatrix.reshape(-1, dmatrix.shape[1] // 10, 10).mean(axis=2)

        dmatrix = pd.DataFrame(
            dmatrix,
            columns=[
                100 * i for i in range(1, dmatrix.shape[1] + 1)
            ] if not agg else [
                1000 * i for i in range(1, dmatrix.shape[1] + 1)
            ]
        )

        return dmatrix

    model_matcher = {
        "ARF-abs": "ARF",
        "XT": "XT"
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
        mid = min_ + (max_ - min_) / 2
        ax = sns.heatmap(
            dmatrix, linewidth=0.05, yticklabels=list(reversed(n_trees)),
            xticklabels=10, cbar_kws={"label": "RMSE"},
            ax=ax, center=mid, vmin=min_, vmax=max_
        )
        for i in range(dmatrix.shape[0] + 1):
            ax.axhline(i, color='white', lw=1)

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

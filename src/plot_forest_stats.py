import copy
import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import DATASETS, OUT_PATH


if not os.path.exists(f"{OUT_PATH}/forest_stats/plots"):
    os.makedirs(f"{OUT_PATH}/forest_stats/plots")

# Just to infer the number of trees per forest
N_DESC_FEATS_TREES = 7
prefixes = ["n_nodes", "n_leaves", "height", "total_observed_weight"]


def plot_saturation_curves(dataset_name):
    m_rf = pd.read_csv(
        f"{OUT_PATH}/forest_stats/final/mean_{dataset_name}_ARF-abs.csv"
    )
    s_rf = pd.read_csv(
        f"{OUT_PATH}/forest_stats/final/std_{dataset_name}_ARF-abs.csv"
    )
    
    m_xt = pd.read_csv(
        f"{OUT_PATH}/forest_stats/final/mean_{dataset_name}_XT.csv"
    )
    s_xt = pd.read_csv(
        f"{OUT_PATH}/forest_stats/final/std_{dataset_name}_XT.csv"
    )

    n_trees = (m_rf.shape[1] - 1) / N_DESC_FEATS_TREES
    instances = m_rf[list(m_rf)[0]]

    m_rf = m_rf.groupby(m_rf.columns.str.extract("(\w+)(?=_\d\d\d)", expand=False), axis=1).mean()
    s_rf = s_rf.groupby(s_rf.columns.str.extract("(\w+)(?=_\d\d\d)", expand=False), axis=1).std()
    m_xt = m_xt.groupby(m_xt.columns.str.extract("(\w+)(?=_\d\d\d)", expand=False), axis=1).mean()
    s_xt = s_xt.groupby(s_xt.columns.str.extract("(\w+)(?=_\d\d\d)", expand=False), axis=1).std()

    fig, ax = plt.subplots(figsize=(5, 10), nrows=4, dpi=600)
    
    for ax_id, prefix in enumerate(prefixes):
    
        ax[ax_id].plot(instances, m_rf[prefix], c="blue", marker="s", label="ARF", markevery=0.1)
        ax[ax_id].plot(instances, m_xt[prefix], c="black", marker="X", label="XT", markevery=0.1)
        
        ax[ax_id].fill_between(
            instances, m_rf[prefix] - s_rf[prefix], m_rf[prefix] + s_rf[prefix],
            color='blue', alpha=0.2
        )
        ax[ax_id].fill_between(
            instances, m_xt[prefix] - s_xt[prefix], m_xt[prefix] + s_xt[prefix],
            color='black', alpha=0.2
        )

    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].xaxis.set_ticklabels([])

    ax[3].set_xlabel("Instances")
    ax[0].set_ylabel("Nodes")
    ax[1].set_ylabel("Leaves")
    ax[2].set_ylabel("Tree height")
    ax[3].set_ylabel("Total resampling weight")

    ax[0].legend()
    ax[0].set_title(f"ARF x XT: tree properties")
    for tick in ax[3].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")


    plt.savefig(
        f"{OUT_PATH}/forest_stats/plots/{dataset_name}.png",
        bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    for dataset in DATASETS:
        plot_saturation_curves(dataset)


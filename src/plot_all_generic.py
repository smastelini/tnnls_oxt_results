import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import DATASETS, OUT_PATH, MODELS

if not os.path.exists(f"{OUT_PATH}/plots"):
    os.makedirs(f"{OUT_PATH}/plots")

color_mapper = {
    "AMRules": "green",
    "ARF-abs": "blue",
    "HT": "red",
    "XT": "black"
}

marker_mapper = {
    "AMRules": "o",
    "ARF-abs": "s",
    "HT": "^",
    "XT": "X"
}

name_prettifier = {
    "AMRules": "AMRules",
    "ARF-abs": "ARF",
    "HT": "HT",
    "XT": "OXT"
}


def plot_curves(dataset_name):
    print(dataset_name)
    fig, ax = plt.subplots(figsize=(10, 5), nrows=2, ncols=2, dpi=600)
    for model_name in MODELS:
        print(f"\t{model_name}")
        mean_data = pd.read_csv(
            f"{OUT_PATH}/final/mean_{dataset_name}_{model_name}.csv"
        )
        std_data = pd.read_csv(
            f"{OUT_PATH}/final/std_{dataset_name}_{model_name}.csv"
        )    
        ax[0][0].plot(
            mean_data["n_samples"], mean_data["RMSE"], 
            color=color_mapper[model_name], marker=marker_mapper[model_name],
            label=name_prettifier[model_name],
            markevery=0.2, linewidth=0.8
        )
        ax[1][0].plot(
            mean_data["n_samples"], mean_data["R2"],
            color=color_mapper[model_name], marker=marker_mapper[model_name],
            markevery=0.2, linewidth=0.8
        )
        ax[0][1].plot(
            mean_data["n_samples"], mean_data["memory"],
            color=color_mapper[model_name], marker=marker_mapper[model_name],
            markevery=0.2, linewidth=0.8
        )
        ax[1][1].plot(
            mean_data["n_samples"], mean_data["time"],
            color=color_mapper[model_name], marker=marker_mapper[model_name],
            markevery=0.2, linewidth=0.8
        )
        

        ax[0][0].fill_between(
            mean_data["n_samples"], mean_data["RMSE"] - std_data["RMSE"], mean_data["RMSE"] + std_data["RMSE"],
                color=color_mapper[model_name], alpha=0.2
        )
        
        ax[1][0].fill_between(
            mean_data["n_samples"], mean_data["R2"] - std_data["R2"], mean_data["R2"] + std_data["R2"],
                color=color_mapper[model_name], alpha=0.2
        )
        ax[0][1].fill_between(
            mean_data["n_samples"], mean_data["memory"] - std_data["memory"], mean_data["memory"] + std_data["memory"],
                color=color_mapper[model_name], alpha=0.2
        )
        ax[1][1].fill_between(
            mean_data["n_samples"], mean_data["time"] - std_data["time"], mean_data["time"] + std_data["time"],
                color=color_mapper[model_name], alpha=0.2
        )

    ax[0][0].xaxis.set_ticklabels([])
    ax[0][1].xaxis.set_ticklabels([])

    ax[1][0].set_xlabel("Instances")
    ax[1][1].set_xlabel("Instances")
    ax[0][0].set_ylabel("RMSE")
    ax[1][0].set_ylabel("R$^2$")
    ax[0][1].set_ylabel("Memory (MB)")
    ax[1][1].set_ylabel("Time (s)")

    ax[0][0].legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.1),
        ncol=2, fancybox=True, shadow=True
    )
    for tick in ax[1][0].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    
    for tick in ax[1][1].get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    
    # plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()

    plt.savefig(
        f"{OUT_PATH}/plots/{dataset_name}.png",
        bbox_inches="tight"
    )
    plt.close()
    

if __name__ == "__main__":
    for dataset_name in DATASETS:
        plot_curves(dataset_name)

    
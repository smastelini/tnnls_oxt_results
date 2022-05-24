import os
import datetime
import pandas as pd

from utils import DATASETS, N_REPS, OUT_PATH


if not os.path.exists(f"{OUT_PATH}/forest_stats/final"):
    os.makedirs(f"{OUT_PATH}/forest_stats/final")


def parse_all(dataset_name):
    for model_name in ["ARF-abs", "XT"]:
        logs = []
        for rep in range(N_REPS):
            logs.append(
                pd.read_csv(
                    f"{OUT_PATH}/forest_stats/{dataset_name}_{model_name}_rep{rep:02}.csv"
                )
            )

            concat = pd.concat(logs)
            concat.groupby(concat.index).mean().to_csv(
                f"{OUT_PATH}/forest_stats/final/mean_{dataset_name}_{model_name}.csv", index=False
            )
            concat.groupby(concat.index).std().to_csv(
                f"{OUT_PATH}/forest_stats/final/std_{dataset_name}_{model_name}.csv", index=False
            )


for dataset_name in DATASETS:
    parse_all(dataset_name)


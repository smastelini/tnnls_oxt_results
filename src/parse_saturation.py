import os
import datetime
import pandas as pd

from utils import N_REPS, OUT_PATH

from parse_outputs import parse_file


if not os.path.exists(f"{OUT_PATH}/saturation/final"):
    os.makedirs(f"{OUT_PATH}/saturation/final")


def parse_all(dataset_name):
    for model_name in ["ARF-abs", "XT"]:
        for t in range(10, 110, 10):
            logs = []
            for rep in range(N_REPS):
                logs.append(
                    parse_file(
                        f"{OUT_PATH}/saturation/results_{dataset_name}_{model_name}_t{t}_rep{rep:02}.txt"
                    )
                )

            concat = pd.concat(logs)
            concat.groupby(concat.index).mean().to_csv(
                f"{OUT_PATH}/saturation/final/mean_{dataset_name}_{model_name}_t{t}.csv", index=False
            )
            concat.groupby(concat.index).std().to_csv(
                f"{OUT_PATH}/saturation/final/std_{dataset_name}_{model_name}_t{t}.csv", index=False
            )


for dataset_name in ["cal_housing", "elevators", "friedman", "friedman_lea", "friedman_gra", "friedman_gsg"]:
    parse_all(dataset_name)


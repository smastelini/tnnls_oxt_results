import os
import datetime
import pandas as pd

from utils import DATASETS, MODELS, N_REPS, OUT_PATH


if not os.path.exists(f"{OUT_PATH}/final"):
    os.makedirs(f"{OUT_PATH}/final")


def parse_file(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

        data = {}
        for i, line in enumerate(lines):
            text = line.replace("\n", "").replace("[", "").split(" – ")
            n_samples, metrics = text[0].split("] ")
            n_samples = int(n_samples.replace(",", ""))

            try:
                runtime = (
                    datetime.datetime.strptime(
                        text[1],
                        "%H:%M:%S.%f"
                    ) - datetime.datetime(1900, 1, 1)
                ).total_seconds()
            except ValueError:
                try:
                    runtime = (
                        datetime.datetime.strptime(
                            text[1],
                            "%H:%M:%S" 
                        ) - datetime.datetime(1900, 1, 1)
                    ).total_seconds()
                except ValueError:
                    n_days, remaining = text[1].split(", ")
                    runtime = datetime.datetime.strptime(remaining,"%H:%M:%S.%f") + datetime.timedelta(days=int(n_days.split(" ")[0]))
                    runtime = (runtime - datetime.datetime(1900, 1, 1)).total_seconds()
            raw_memory, unit = text[2].split(" ")
            memory = float(raw_memory) if unit == "MB" else float(raw_memory) / (2 ** 10)

            data[i] = {
                "n_samples": n_samples,
                "time": runtime,
                "memory": memory
            }
            data[i].update(
                dict(
                    map(
                        lambda raw_metric: (raw_metric[0], float(raw_metric[1].replace(",", ""))),
                        map(
                            lambda raw_metric: raw_metric.split(": "),
                            metrics.split(", ")
                        )
                    )
                )
            )

        return pd.DataFrame.from_dict(data, orient="index")


def parse_all(dataset_name):
    for model_name in MODELS:
        logs = []
        for rep in range(N_REPS):
            logs.append(
                parse_file(
                    f"{OUT_PATH}/results_{dataset_name}_{model_name}_rep{rep:02}.txt"
                )
            )

        concat = pd.concat(logs)
        concat.groupby(concat.index).mean().to_csv(
            f"{OUT_PATH}/final/mean_{dataset_name}_{model_name}.csv", index=False
        )
        concat.groupby(concat.index).std().to_csv(
            f"{OUT_PATH}/final/std_{dataset_name}_{model_name}.csv", index=False
        )

if __name__ == "__main__":
    for dataset_name in DATASETS:
        parse_all(dataset_name)


import copy
import random

from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from river import synth

from utils import DRIFT_DATASETS, IN_PATH, MAIN_SEED, MODELS, N_REPS, OUT_PATH


def get_drift_data(name):
    if name == "friedman_lea":
        return synth.FriedmanDrift(
            drift_type="lea",
            position=(25_000, 50_000, 75_000),
            seed=7
        ).take(100_000)
    elif name == "friedman_gra":
        return synth.FriedmanDrift(
            drift_type="gra",
            position=(35_000, 75_000),
            seed=7,
        ).take(100_000)
    elif name == "friedman_gsg":
        return synth.FriedmanDrift(
            drift_type="gsg",
            position=(35_000, 75_000),
            seed=7,
            transition_window=2_000
        ).take(100_000)
    else:
        header = ""
        with open(f"{IN_PATH}/{name}.csv", "r") as f:
            header = f.readline()

        input_names = header.replace("\n", "").split(",")
        target = input_names[-1]

        dataset = stream.iter_csv(
            f"{IN_PATH}/{name}.csv",
            target=target,
            converters={
                input_name: float for input_name in input_names
            }
        )

        return dataset


def run_dataset(dataset_name):
    print(dataset_name)
    rng = random.Random(MAIN_SEED)
    seeds = [rng.randint(0, 99999) for _ in range(N_REPS)]
    for model_name, template_model in MODELS.items():
        print(model_name, end="\t")
        for rep in range(N_REPS):
            print(f" {rep}", end="")
            dataset = get_drift_data(dataset_name)

            # Change seed if the model supports it
            model = copy.deepcopy(template_model)._set_params({"seed": seeds[rep]})

            # Assemble the final model
            model = preprocessing.StandardScaler() | model

            evaluate.progressive_val_score(
                dataset,
                model,
                metrics.MAE() + metrics.RMSE() + metrics.R2(),
                show_memory=True,
                show_time=True,
                print_every=100,
                file=open(
                    f"{OUT_PATH}/results_{dataset_name}_{model_name}_rep{rep:02d}.txt",
                    "w"
                )
            )
        print()


for dataset_name in DRIFT_DATASETS:
    run_dataset(dataset_name)


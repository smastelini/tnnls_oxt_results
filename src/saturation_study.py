import os
import copy
import math
import numbers
import random

from river import compose
from river import evaluate
from river import metrics
from river import preprocessing
from river import stream
from river import synth

from utils import CAT_FEATURES, DATASETS, GENERATOR_BASED, IN_PATH, MAIN_SEED, MODELS, N_REPS, OUT_PATH


def prepare_data(dataset_name, seed=None):
    if dataset_name not in GENERATOR_BASED:
        header = ""
        with open(f"{IN_PATH}/{dataset_name}.csv", "r") as f:
            header = f.readline()

        names = header.replace("\n", "").split(",")
        features, target = names[:-1], names[-1]

        if dataset_name in CAT_FEATURES:
            nominal_attributes = [features[i] for i in CAT_FEATURES[dataset_name]]
        else:
            nominal_attributes = None

        dataset = stream.iter_csv(
            f"{IN_PATH}/{dataset_name}.csv",
            target=target,
            converters={
                name: float for name in names
            }
        )

        dataset = stream.shuffle(
            dataset,
            buffer_size=100,
            seed=seed
        )
    else:
        nominal_attributes = None
        if dataset_name == "friedman":
            dataset = synth.Friedman(seed=seed).take(100_000)
        elif dataset_name == "planes2d":
            dataset = synth.Planes2D(seed=seed).take(100_000)
        elif dataset_name == "mv":
            dataset = synth.Mv(seed=seed).take(100_000)
        if dataset_name == "friedman_lea":
            dataset = synth.FriedmanDrift(
                drift_type="lea",
                position=(25_000, 50_000, 75_000),
                seed=seed
            ).take(100_000)
        elif dataset_name == "friedman_gra":
            dataset = synth.FriedmanDrift(
                drift_type="gra",
                position=(35_000, 75_000),
                seed=seed,
            ).take(100_000)
        elif dataset_name == "friedman_gsg":
            dataset = synth.FriedmanDrift(
                drift_type="gsg",
                position=(35_000, 75_000),
                seed=seed,
                transition_window=2_000
            ).take(100_000)

    return dataset, nominal_attributes


def saturate(model_name, dataset_name, init_models=10, final_models=100, incr=10):
    print(dataset_name)
    rng = random.Random(MAIN_SEED)
    seeds = [rng.randint(0, 99999) for _ in range(N_REPS)]
    for n_models in range(init_models, final_models + incr, incr):
        print(f"Number of trees: {n_models}")
        for rep in range(N_REPS):
            print(f"Rep {rep}")
            dataset, nominal_attributes = prepare_data(dataset_name, seed=seeds[rep])
            model = copy.deepcopy(MODELS[model_name])
            params = model._get_params()
            params["n_models"] = n_models
            model = model._set_params(params)

            if dataset_name in CAT_FEATURES:
                model.nominal_attributes = nominal_attributes
                preproc = (
                    (compose.Discard(*tuple(nominal_attributes)) | compose.SelectType(
                        numbers.Number) | preprocessing.StandardScaler()) + compose.Select(
                        *tuple(nominal_attributes)) + compose.SelectType(str)
                )
            else:
                preproc = (
                    (
                        compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
                    ) + compose.SelectType(str)
                )

            # Assemble the final model
            model = preproc | model

            evaluate.progressive_val_score(
                dataset,
                model,
                metrics.MAE() + metrics.RMSE() + metrics.R2(),
                show_memory=True,
                show_time=True,
                print_every=100,
                file=open(
                    f"{OUT_PATH}/saturation/results_{dataset_name}_{model_name}_t{n_models}_rep{rep:02d}.txt",
                    "w"
                )
            )


if not os.path.exists(f"{OUT_PATH}/saturation"):
    os.makedirs(f"{OUT_PATH}/saturation")


#saturate("ARF-abs", "friedman_lea", init_models=90, final_models=100)
#saturate("XT", "friedman_gsg")
#saturate("ARF-abs-mean", "friedman_gra")
# saturate("XT", "friedman_gra")
saturate("XT-mean", "friedman_gra")



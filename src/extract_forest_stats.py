import os
import copy
import numbers
import random

import pandas as pd

from river import compose
from river import preprocessing
from river import stream
from river import synth
#from river.utils import warm_up_mode


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


def extract_measures(dataset, model):
    log = {}
    #with warm_up_mode():
    for i, (x, y) in enumerate(dataset):
        # We are not interested on predictions here, only the model structure
        model.learn_one(x, y)
        
        if (i + 1) % 100 == 0:
            row = {}
            forest = model["forest"]
            for n in range(forest.n_models):
                aux = {
                    f"{s_n}_{n:03d}": s_v for s_n, s_v in forest[n].summary.items()
                }
                row.update(aux)
                
                
            log[i + 1] = row
      
    return pd.DataFrame.from_dict(log, orient="index")


def run_reps(dataset_name, model_name):
    print(dataset_name)
    rng = random.Random(MAIN_SEED)
    seeds = [rng.randint(0, 99999) for _ in range(N_REPS)]
    for rep in range(N_REPS):
        print(f"Rep {rep}")
        dataset, nominal_attributes = prepare_data(dataset_name, seed=seeds[rep])
        model = copy.deepcopy(MODELS[model_name])

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
        model = compose.Pipeline(
            ("preproc", preproc),
            ("forest", model)
        )
         
        log = extract_measures(dataset, model)
        log.to_csv(
            f"{OUT_PATH}/forest_stats/{dataset_name}_{model_name}_rep{rep:02d}.csv",
        )


if __name__ == "__main__":
    if not os.path.exists(f"{OUT_PATH}/forest_stats"):
        os.makedirs(f"{OUT_PATH}/forest_stats")
        
    for model_name in ["XT"]:
        for dataset_name in DATASETS:
            run_reps(dataset_name, model_name)


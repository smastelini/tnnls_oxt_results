import sys

from river import drift
from river import dummy
from river import ensemble
from river import linear_model
from river import metrics
from river import rules
from river import stats
from river import tree


from river_extra.ensemble import ExtraTreesRegressor


IN_PATH = "../datasets"
# OUT_PATH = "/lustre/smm/online-learning/output_xt_eager_split"
OUT_PATH = "../output"

MAIN_SEED = 42
INCLUDE_STD_IN_TABLES = True
INCLUDE_BASELINES_IN_TABLES = False
INCLUDE_DRIFT_DATASETS_IN_TABLES = True


DATASETS = {
    "abalone": "Abalone",
    "ailerons": "Ailerons",
    # "airlines07_08": "Airlines 07-08",
    "bike": "Bike",
    "cal_housing": "CalHousing",
    "elevators": "Elevators",
    "house_8L": "House8L",
    "house_16H": "House16H",
    "metro_interstate_traffic": "Metro",
    # "msd_year": "MSD Year",
    "pol": "Pol",
    "wind": "Wind",
    "winequality": "Wine",
    # # synthetic
    "friedman": "Friedman",
    # "mv": "MV",
    "puma8NH": "Puma8NH",
    "puma32H": "Puma32H",
}


DRIFT_DATASETS = {
    "friedman_lea": "Friedman(LEA)",
    "friedman_gra": "Friedman(GRA)",
    "friedman_gsg": "Friedman(GSG)",
    "hyper_reg_3abrupt_i500k": "Hyper(A)",
    "hyper_reg_3gradual_i500k": "Hyper(G)",
    "hyper_t001_k4_reg_incremental_i500k": "Hyper(I)",
    "rbf_reg_3abrupt_i500k": "RBF(A)",
    "rbf_reg_3gradual_i500k": "RBF(G)",
    # "rbf_reg_incremental_i500k": "RBF(I)"
}


SYNTH_DATA = {
    "friedman",
    "mv",
    "puma8NH",
    "puma32H",
    "friedman_lea",
    "friedman_gra",
    "friedman_gsg",
    "hyper_reg_3abrupt_i500k",
    "hyper_reg_3gradual_i500k",
    "hyper_t001_k4_reg_incremental_i500k",
    "rbf_reg_3abrupt_i500k",
    "rbf_reg_3gradual_i500k",
    # "rbf_reg_incremental_i500k"
}

GENERATOR_BASED = {
    "friedman",
    "mv",
    "friedman_lea",
    "friedman_gra",
    "friedman_gsg"
}

CAT_FEATURES = {
    "abalone": [0],
    "airlines07_08": [1, 2, 3, 6, 9, 10, 12],
    "metro_interstate_traffic": [0, 5, 6],
    "wind": [1, 2]
}

N_REPS = 5


MODELS = {
    "HT": tree.HoeffdingTreeRegressor(
        splitter=tree.splitter.TEBSTSplitter(digits=1),
        leaf_prediction="adaptive"
    ),
    #"HAT": tree.HoeffdingAdaptiveTreeRegressor(
    #    splitter=tree.splitter.TEBSTSplitter(digits=1),
    #    leaf_prediction="adaptive",
    #    seed=42,
    #    bootstrap_sampling=True
    #),
    "AMRules": rules.AMRules(
        drift_detector=drift.ADWIN(),
        splitter=tree.splitter.TEBSTSplitter(digits=1)
    ),
    "ARF-abs": ensemble.AdaptiveRandomForestRegressor(
        n_models=20,
        leaf_prediction="adaptive",
        splitter=tree.splitter.TEBSTSplitter(digits=1),
        seed=42
    ),
    "XT": ExtraTreesRegressor(
        n_models=20,
        seed=42,
        track_metric=metrics.RMSE(),
        resampling_strategy="subbagging",
        resampling_rate=0.5,
        disable_weighted_vote=False,
        leaf_prediction="adaptive",
        split_confidence=0.05,
        merit_preprune=False,
    ),
    # Using the mean prediction
    #"ARF-abs-mean": ensemble.AdaptiveRandomForestRegressor(
    #    n_models=20,
    #    leaf_prediction="mean",
    #    splitter=tree.splitter.TEBSTSplitter(digits=1),
    #    seed=42
    #),
    # "XT-mean": ExtraTreesRegressor(
    #    n_models=20,
    #     seed=42,
    #     track_metric=metrics.RMSE(),
    #     resampling_strategy="subbagging",
    #     resampling_rate=0.5,
    #     disable_weighted_vote=False,
    #     leaf_prediction="mean",
    #     split_confidence=0.05,
    #     merit_preprune=False,
    # ),
}


BASELINES = {
    "PAR": linear_model.PARegressor(),
    "Dummy": dummy.StatisticRegressor(stats.Mean()),
}

if INCLUDE_BASELINES_IN_TABLES:
    BASELINES.update(MODELS)
    MODELS = BASELINES

if INCLUDE_DRIFT_DATASETS_IN_TABLES:
    DATASETS.update(DRIFT_DATASETS)

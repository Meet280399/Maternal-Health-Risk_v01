from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from df_analyze.analysis.univariate.associate import target_associations
from df_analyze.analysis.univariate.predict.predict import univariate_predictions
from df_analyze.cli.cli import ProgramOptions
from df_analyze.enumerables import (
    DfAnalyzeClassifier,
    DfAnalyzeRegressor,
    EmbedSelectionModel,
    FeatureSelection,
    WrapperSelection,
    WrapperSelectionModel,
)
from df_analyze.hypertune import evaluate_tuned
from df_analyze.nonsense import silence_spam
from df_analyze.preprocessing.cleaning import sanitize_names
from df_analyze.preprocessing.inspection.inspection import inspect_data
from df_analyze.preprocessing.prepare import prepare_data
from df_analyze.selection.filter import filter_select_features
from df_analyze.selection.models import model_select_features
from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    FASTEST,
    MEDIUM_INSPECTION,
    SLOW_INSPECTION,
    TestDataset,
    fast_ds,
    turbo_ds,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def do_main(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname == "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc":
        return  # target undersampled levels
    options = ProgramOptions.random(ds)
    options.feat_select = (
        FeatureSelection.Filter,
        FeatureSelection.Wrapper,
        FeatureSelection.Embedded,
    )
    options.embed_select = (EmbedSelectionModel.LGBM, EmbedSelectionModel.Linear)
    # options.embed_select = None
    options.wrapper_model = WrapperSelectionModel.Linear
    options.wrapper_select = WrapperSelection.StepUp
    # options.wrapper_select = None
    options.n_feat_wrapper = 10

    if ds.is_classification:
        options.classifiers = (
            DfAnalyzeClassifier.Dummy,
            DfAnalyzeClassifier.KNN,
            DfAnalyzeClassifier.SGD,
            DfAnalyzeClassifier.LR,
            DfAnalyzeClassifier.MLP,
            DfAnalyzeClassifier.LGBM,
            # DfAnalyzeClassifier.SVM,
        )
        # options.classifiers = (DfAnalyzeClassifier.MLP,)
    else:
        options.regressors = (
            DfAnalyzeRegressor.Dummy,
            DfAnalyzeRegressor.KNN,
            DfAnalyzeRegressor.SGD,
            DfAnalyzeRegressor.ElasticNet,
            DfAnalyzeRegressor.MLP,
            DfAnalyzeRegressor.LGBM,
            # DfAnalyzeRegressor.SVM,
        )
        # options.regressors = (DfAnalyzeRegressor.MLP,)
    options.htune_trials = 20
    options.to_json()

    is_cls = options.is_classification
    prog_dirs = options.program_dirs
    target = options.target
    categoricals = options.categoricals
    ordinals = options.ordinals
    drops = options.drops
    grouper = options.grouper

    df = options.load_df()
    df, renames = sanitize_names(df, target)
    prog_dirs.save_renames(renames)

    categoricals = renames.rename_columns(categoricals)
    ordinals = renames.rename_columns(ordinals)
    drops = renames.rename_columns(drops)

    df, inspection = inspect_data(
        df=df,
        target=target,
        grouper=grouper,
        categoricals=categoricals,
        ordinals=ordinals,
        drops=drops,
        _warn=True,
    )
    prog_dirs.save_inspect_reports(inspection)
    prog_dirs.save_inspect_tables(inspection)

    prepared = prepare_data(
        df=df,
        target=target,
        grouper=grouper,
        results=inspection,
        is_classification=is_cls,
    )
    prog_dirs.save_prepared_raw(prepared)
    prog_dirs.save_prep_report(prepared.to_markdown())
    prep_train, prep_test = prepared.split()

    associations = target_associations(prep_train)
    prog_dirs.save_univariate_assocs(associations)
    prog_dirs.save_assoc_report(associations.to_markdown())

    predictions = univariate_predictions(prep_train, is_cls)
    prog_dirs.save_univariate_preds(predictions)
    prog_dirs.save_pred_report(predictions.to_markdown())

    # select features via filter methods first
    assoc_filtered, pred_filtered = filter_select_features(
        prep_train, associations, predictions, options
    )
    prog_dirs.save_filter_report(assoc_filtered)
    prog_dirs.save_filter_report(pred_filtered)

    # TODO: make embedded and wrapper selection mutually exclusive. Only two
    # phases of feature selection: filter selection, and model-based
    # selection, wher model-based selection means either embedded or wrapper
    # (stepup, stepdown) methods.
    selected = model_select_features(prep_train, options)
    # selected = ModelSelected.random(ds)
    prog_dirs.save_model_selection_reports(selected)
    prog_dirs.save_model_selection_data(selected)

    silence_spam()
    eval_results = evaluate_tuned(
        prepared=prepared,
        prep_train=prep_train,
        prep_test=prep_test,
        assoc_filtered=assoc_filtered,
        pred_filtered=pred_filtered,
        model_selected=selected,
        options=options,
    )
    print(eval_results.to_markdown())
    prog_dirs.save_eval_report(eval_results)
    prog_dirs.save_eval_tables(eval_results)
    prog_dirs.save_eval_data(eval_results)

    # prog_dirs.save_final_reports(results)
    # prog_dirs.save_final_tables(results)
    # prog_dirs.save_final_plots(results)

    """
    if options.embedded_select:
        selected

    """


@turbo_ds
def test_main_turbo(dataset: tuple[str, TestDataset]) -> None:
    do_main(dataset)


@fast_ds
def test_main_fast(dataset: tuple[str, TestDataset]) -> None:
    do_main(dataset)


if __name__ == "__main__":
    # for dsname, ds in FASTEST[:1]:
    #     print("=" * 79)
    #     print(f"Testing {dsname}")
    #     print("=" * 79)
    #     do_main((dsname, ds))

    # sys.exit()

    DATASETS = FAST_INSPECTION + MEDIUM_INSPECTION + SLOW_INSPECTION
    if os.environ.get("CC_CLUSTER") == "niagara":
        idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if idx is None:
            raise ValueError("On Niagara but no SLURM_ARRAY_TASK_ID defined")
        idx = int(idx)
        dsname, ds = DATASETS[idx]
        print("=" * 79)
        print(f"Testing {dsname}")
        print("=" * 79)
        do_main((dsname, ds))

    else:
        for dsname, ds in FAST_INSPECTION:
            if "kdd" in dsname.lower():
                continue  # all slow as hell
            # if dsname.lower() < "student_dropout":
            #     continue

            print("=" * 79)
            print(f"Testing {dsname}")
            print("=" * 79)
            do_main((dsname, ds))

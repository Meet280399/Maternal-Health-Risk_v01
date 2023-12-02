from math import ceil
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from dateutil.parser import parse
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from src.enumerables import NanHandling
from src.loading import load_spreadsheet
from src.preprocessing.inspection import inspect_str_columns, looks_timelike


class DataCleaningWarning(UserWarning):
    """For when df-analyze significantly and automatically alters input data"""


def normalize(df: DataFrame, target: str) -> DataFrame:
    """
    MUST be min-max normalization (e.g. for stat tests with no negatives)
    and on the categorical-encoded df.
    """
    X = df.drop(columns=target)
    X_norm = DataFrame(data=MinMaxScaler().fit_transform(X), columns=X.columns)
    X_norm[target] = df[target]
    return X_norm


def handle_continuous_nans(
    df: DataFrame, target: str, cat_cols: list[str], nans: NanHandling
) -> DataFrame:
    """Impute or drop nans based on values not in `cat_cols`"""
    # drop rows where target is NaN: meaningless
    idx = ~df[target].isna()
    df = df.loc[idx]
    # NaNs in categoricals are handled as another dummy indicator
    drops = list(set(cat_cols).union([target]))
    X = df.drop(columns=drops, errors="ignore")
    X_cat = df[cat_cols]
    y = df[target]

    if nans is NanHandling.Drop:
        X_clean = X.dropna(axis="columns").dropna(axis="index")
        if 0 in X_clean.shape:
            others = [na.value for na in NanHandling if na is not NanHandling.Drop]
            raise RuntimeError(
                "Dropping NaNs resulted in either no remaining samples or features. "
                "Consider changing the option `--nan` to another valid option. Other "
                f"valid options: {others}"
            )

    elif nans in [NanHandling.Mean, NanHandling.Median]:
        strategy = "mean" if nans is NanHandling.Mean else "median"
        X_clean = DataFrame(
            data=SimpleImputer(strategy=strategy).fit_transform(X), columns=X.columns
        )
    elif nans is NanHandling.Impute:
        warn(
            "Using experimental multivariate imputation. This could take a very "
            "long time for even tiny (<500 samples, <50 features) datasets."
        )
        imputer = IterativeImputer(verbose=2)
        X_clean = DataFrame(data=imputer.fit_transform(X), columns=X.columns)
    else:
        raise NotImplementedError(f"Unhandled enum case: {nans}")

    X_clean[target] = y

    return pd.concat([X_cat, X_clean], axis=1)


def encode_target(df: DataFrame, target: Series) -> tuple[DataFrame, Series]:
    unqs, cnts = np.unique(target, return_counts=True)
    idx = cnts <= 20
    n_cls = len(unqs)
    if np.sum(idx) > 0:
        warn(
            f"The target variable has a number of class labels ({unqs[idx]}) with "
            "less than 20 members. This will cause problems with splitting in "
            "various nested k-fold procedures used in `df-analyze`. In addition, "
            "any estimates or metrics produced for such a class will not be "
            "statistically meaningful (i.e. the uncertainty on those metrics or "
            "estimates will be exceedingly large). We thus remove all samples "
            "that belong to these labels, bringing the total number of classes "
            f"down to {n_cls - np.sum(idx)}",
            category=DataCleaningWarning,
        )
        idx_drop = ~target.isin(unqs[idx])
        df = df.copy().loc[idx_drop]
        target = target[idx_drop]

    # drop NaNs: Makes no sense to count correct NaN predictions toward
    # classification performance
    idx_drop = ~target.isna()
    df = df.copy().loc[idx_drop]
    target = target[idx_drop]

    encoded = np.array(LabelEncoder().fit_transform(target))
    return df, Series(encoded, name=target.name)


def clean_regression_target(df: DataFrame, target: Series) -> tuple[DataFrame, Series]:
    """NaN targets cannot be predicted. Remove them, and then robustly
    normalize target to facilitate convergence and interpretation
    of metrics
    """
    idx_drop = ~target.isna()
    df = df.loc[idx_drop]
    target = target[idx_drop]

    y = (
        RobustScaler(quantile_range=(2.5, 97.5))
        .fit_transform(target.to_numpy().reshape(-1, 1))
        .ravel()
    )
    target = Series(y, name=target.name)

    return df, target


def drop_id_cols(df: DataFrame, target: str) -> tuple[DataFrame, list[str]]:
    X = df.drop(columns=target, errors="ignore").infer_objects()
    unique_counts = {}
    for colname in X.columns:
        try:
            unique_counts[colname] = len(np.unique(df[colname]))
        except TypeError:  # happens when can't sort for unique
            unique_counts[colname] = len(np.unique(df[colname].astype(str)))
    str_cols = X.select_dtypes(include=["object", "string[python]"]).columns.tolist()
    # Reasoning: if there are more levels in a categorical than 1/5 of the number
    # of samples, then in 5-fold, test set will have most levels effectively never
    # seen before (unless distribution of levels is highly skwewed).
    id_cols = [col for col in str_cols if unique_counts[col] >= len(df) // 5]
    if len(id_cols) == 0:
        return df, []

    warn(
        r"Found string-valued features with more unique levels than 20% of "
        "the total number of samples in the data. This is most likely an "
        "'identifier' or junk feature which has no predictive value, and most "
        "likely should be removed from the data. Even if this is not the case, "
        "with such a large number of levels, then a test set (either in k-fold, "
        "or holdout) will likely simply contain a large number of values for "
        "such a categorical variable that were never seen during training. Thus "
        "these features are likely too sparse to be useful given the amount of data, "
        "and also massively increase compute costs for likely no gain. We thus "
        "REMOVE these features and do not one-hot encode them. To silence this "
        "warning, either remove these features from the data, or manually break "
        "them into a smaller number of categories. "
        f"String-valued features with too many levels: {id_cols}"
    )
    X = X.drop(columns=id_cols)
    X[target] = df[target]
    return X, id_cols


def detect_big_cats(unique_counts: dict[str, int], str_cols: list[str]) -> list[str]:
    big_cats = [col for col in str_cols if (col in unique_counts) and (unique_counts[col] >= 20)]
    if len(big_cats) > 0:
        warn(
            "Found string-valued features with more than 50 unique levels. "
            "Unless you have an extremely large number of samples, or if these "
            "features have a highly imbalanced / skewed distribution, then they "
            "will cause sparseness after one-hot encoding. This is generally not "
            "beneficial to most algorithms. You should inspect these features and "
            "think if it makes sense if they would be predictively useful for the "
            "given target. If they are unlikely to be useful, consider removing "
            "them from the data. This will also likely considerably improve "
            "`df-analyze` predictive performance and reduce compute times. "
            f"String-valued features with over 50 levels: {big_cats}"
        )
    return big_cats


def detect_sus_cols(
    unique_counts: dict[str, int], int_cols: list[str], cats: Union[list[str], int], warn_sus: bool
) -> list[str]:
    sus_ints = [col for col in int_cols if unique_counts[col] < 5]
    if isinstance(cats, list):
        sus_cols = sorted(set(sus_ints).difference(cats))
    else:
        sus_cols = sorted(set(sus_ints))
    if len(sus_cols) > 0 and warn_sus:
        warn(
            "Found integer-valued features not specified with `--categoricals` "
            "argument, and which have less than 5 levels (classes) each. "
            "These features may in fact be categoricals, and if so, should be "
            "specified as such manually either via the CLI or in the "
            "spreadsheet file header. Otherwise, they will be left as is, and "
            "treated as ordinal / continuous. Categorical-like integer-valued "
            f"features: {sus_cols}"
        )
    return sus_cols


def get_unq_counts(df: DataFrame, target: str) -> dict[str, int]:
    X = df.drop(columns=target, errors="ignore").infer_objects().convert_dtypes()
    unique_counts = {}
    for colname in X.columns:
        try:
            unique_counts[colname] = len(np.unique(df[colname]))
        except TypeError:  # happens when can't sort for unique
            unique_counts[colname] = len(np.unique(df[colname].astype(str)))
    return unique_counts


def floatify(df: DataFrame) -> DataFrame:
    df = df.copy()
    cols = df.select_dtypes(include=["object", "string[python]"]).columns.tolist()
    for col in cols:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


def get_cat_cols(df: DataFrame, target: str, categoricals: Union[list[str], int]) -> list[str]:
    """
    Parameters
    ----------
    df: DataFrame
        Data. Must contain target.

    target: str
        Target column name.

    categoricals: Union[list[str], int]
        User-provided CLI argument to `--categoricals`.
    """
    df, drops = drop_id_cols(df, target)

    unique_counts = get_unq_counts(df, target)
    X = floatify(df.drop(columns=target, errors="ignore").infer_objects())
    str_cols = X.select_dtypes(include=["object", "string[python]"]).columns.tolist()

    cats = categoricals
    auto = isinstance(cats, int)
    # if no threshold specified, only convert what absolutely has to be converted
    threshold = cats if auto else -1
    if auto:
        cat_cols = [col for col, cnt in unique_counts.items() if cnt <= threshold]
        cat_cols += str_cols
    else:
        # warn the user if some int columns look sus, and check for unspecified
        # categoricals
        assert isinstance(cats, list)

        cats = set(cats)
        unspecified = sorted(set(str_cols).difference(cats))
        if len(unspecified) > 0:
            warn(
                "Found string-valued features not specified with `--categoricals` "
                "argument. These will be one-hot encoded to allow use in subsequent "
                "analyses. To silence this warning, specify the categoricals "
                "manually either via the CLI or in the spreadsheet file header."
                f"Unspecified string-valued features: {unspecified}"
            )

        cat_cols = list(set(str_cols).union(cats))
    cat_cols = list(set(cat_cols).difference(drops))
    return cat_cols


def encode_categoricals(
    df: DataFrame, target: str, categoricals: Union[list[str], int], _warn: bool = True
) -> tuple[DataFrame, DataFrame]:
    """Treat all features with <= options.cat_threshold as categorical

    Returns
    -------
    encoded: DataFrame
        Pandas DataFrame with categorical variables one-hot encoded

    unencoded: DataFrame
        Pandas DataFrame with original categorical variables

    """
    X = df.drop(columns=target, errors="ignore").infer_objects().convert_dtypes()
    str_cols = X.select_dtypes(include=["object", "string[python]"]).columns.tolist()
    int_cols = X.select_dtypes(include="int").columns.to_list()

    floats, ords, ids, times = inspect_str_columns(X, str_cols, _warn=_warn)

    df, drops = drop_id_cols(df, target)
    if isinstance(categoricals, list):
        categoricals = list(set(categoricals).difference(drops))

    unique_counts = get_unq_counts(df=df, target=target)

    detect_big_cats(unique_counts, str_cols)
    detect_sus_cols(unique_counts, int_cols, categoricals, warn_sus=_warn)

    to_convert = get_cat_cols(df=df, target=target, categoricals=categoricals)
    # below will FAIL if we didn't remove timestamps or etc.
    try:
        new = pd.get_dummies(X, columns=to_convert, dummy_na=True, dtype=float)
        new = new.astype(float)
    except TypeError:
        inspect_str_columns(X, str_cols, _warn=True)
        raise RuntimeError(
            "Could not convert data to floating point after cleaning. Some "
            "messy data must be removed or cleaned before `df-analyze` can "
            "continue. See information above."
        )
    new = pd.concat([new, df[target]], axis=1)
    # new[target] = df[target]
    return new, X.loc[:, to_convert]


def remove_timestamps(df: DataFrame, target: str) -> tuple[DataFrame, list[str]]:
    n_subsamp = max(ceil(0.5 * len(df)), 500)
    n_subsamp = min(n_subsamp, len(df))
    # time checks can be very slow, so just check a few for each feature first
    X = (
        df.drop(columns=target, errors="ignore")
        .infer_objects()
        .select_dtypes(include="object")
        .dropna(axis="index")
    )
    drops = []
    for col in X.columns:
        maybe_time = looks_timelike(X[col])[0]
        if maybe_time:
            drops.append(col)

    return df.drop(columns=drops), drops


def prepare_data(
    df: DataFrame, target: str, categoricals: Union[list[str], int], is_classification: bool
) -> tuple[DataFrame, Series, list[str]]:
    y = df[target]
    df, t_drops = remove_timestamps(df, target)
    df, id_drops = drop_id_cols(df, target)
    cats = []
    if isinstance(categoricals, list):
        cats = list(set(categoricals).difference(t_drops).difference(id_drops))
    cat_cols = get_cat_cols(df=df, target=target, categoricals=cats)
    df = handle_continuous_nans(df=df, target=target, cat_cols=cat_cols, nans=NanHandling.Mean)

    df = df.drop(columns=target)
    if is_classification:
        df, y = encode_target(df, y)
    else:
        df, y = clean_regression_target(df, y)
    return df, y, cat_cols


def load_as_df(path: Path, spreadsheet: bool) -> DataFrame:
    FILETYPES = [".json", ".csv", ".npy", "xlsx", ".parquet"]
    if path.suffix not in FILETYPES:
        raise ValueError(f"Invalid data file. Currently must be one of: {FILETYPES}")
    if path.suffix == ".json":
        df = pd.read_json(str(path))
    elif path.suffix == ".csv":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_csv(str(path))
    elif path.suffix == ".parquet":
        df = pd.read_parquet(str(path))
    elif path.suffix == ".xlsx":
        if spreadsheet:
            df = load_spreadsheet(path)[0]
        else:
            df = pd.read_excel(str(path))
    elif path.suffix == ".npy":
        arr: ndarray = np.load(str(path), allow_pickle=False)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Invalid NumPy data in {path}. NumPy array must be two-dimensional."
            )
        cols = [f"c{i}" for i in range(arr.shape[1])]
        df = DataFrame(data=arr, columns=cols)
    else:
        raise RuntimeError("Unreachable!")
    return df

from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from pprint import pprint
from shutil import get_terminal_size
from sys import stderr
from time import perf_counter

import numpy as np
import pytest
from pandas import DataFrame

from df_analyze.preprocessing.inspection.inspection import (
    get_str_cols,
    inspect_data,
    inspect_str_columns,
    inspect_target,
)
from df_analyze.testing.datasets import (
    TEST_DATASETS,
    TestDataset,
    all_ds,
    fast_ds,
    med_ds,
    slow_ds,
)

TIME_DSNAMES = [
    "elder",
    "forest_fires",
    "soybean",
    "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc",
    "Kaggle_bike_sharing_demand_challange",
    "Insurance",
    "fps_benchmark",
    "bank-marketing",
    "kick",
]
TIME_DATASETS = {
    dsname: ds for dsname, ds in TEST_DATASETS.items() if dsname in TIME_DSNAMES
}


def do_inspect(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    cats = ds.categoricals

    try:
        inspect_data(df=df, target="target", grouper=None, categoricals=cats)
    except TypeError as e:
        if dsname == "community_crime" and (
            "Cannot automatically determine the cardinality" in str(e)
        ):
            return
        raise e
    except Exception as e:
        raise ValueError(f"Could not encode categoricals for data: {dsname}") from e


def do_timestamp_detection(dataset: tuple[str, TestDataset]) -> None:
    """
    False Positives:

      Kaggle_bike_sharing_demand_challange: 'time'
      Insurance: 'Holding_Policy_Duration'
      fps_benchmark: 'GpuOpenCL'
      bank-marketing: 'V11'
      forest_fires: 'day' (is mon, tue, wed, ...)

    True Positives:

      elder: 'timestamp'
      soybean: 'date'
      dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc: 'date_diagnostic'

    Undecided:

    ozone_level: {many}
    kick: {'WheelTypeID': '100% of data parses as datetime'}

    colic: capillary_refill_time
    arrhythmia: QRSduration
    elder: timestamp
    student_performance_por: traveltime
    student_performance_por: studytime
    student_performance_por: freetime
    soybean: date
    dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc: date_plantation
    dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc: date_diagnostic
    pbcseq: prothrombin_time
    student_dropout: daytime/evening attendance
    student_dropout: tuition fees up to date
    colleges: percent_part_time
    colleges: percent_part_time_faculty
    kdd_internet_usage: actual_time
    Kaggle_bike_sharing_demand_challange: time
    OnlineNewsPopularity: timedelta
    OnlineNewsPopularity: global_sentiment_polarity
    OnlineNewsPopularity: title_sentiment_polarity
    OnlineNewsPopularity: abs_title_sentiment_polarity
    news_popularity: global_sentiment_polarity
    news_popularity: title_sentiment_polarity
    news_popularity: abs_title_sentiment_polarity

    """
    dsname, ds = dataset
    df = ds.load()
    str_cols = get_str_cols(df, "target")
    times = inspect_str_columns(df, str_cols, _warn=False)[3]
    assert len(times.infos) == 1


@all_ds
@pytest.mark.wip
def test_str_continuous_warn(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    if dsname not in ["community_crime"]:
        return
    df = ds.load()
    X = df.drop(columns="target")
    dtypes = ["object", "string[python]"]
    cols = X.select_dtypes(include=dtypes).columns.tolist()

    # with pytest.warns(UserWarning, match=".*converted into floating.*"):
    inspect_str_columns(df, str_cols=cols, _warn=False)


def do_detect_floats(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    cats = ds.categoricals
    conts = ds.continuous
    results = inspect_data(df=df, target="target", grouper=None, categoricals=cats)
    float_cols = [*results.conts.infos.keys()]
    if sorted(float_cols) != sorted(conts):
        raise ValueError(f"Columns detected as continuous not as expected for {dsname}")

    X = df[float_cols]
    try:
        X.astype(float)
    except Exception as e:
        raise ValueError(
            f"Columns detected as float for data {dsname} could not be coerced to float"
        ) from e


def do_detect_ids(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    cats = ds.categoricals
    try:
        results = inspect_data(df=df, target="target", grouper=None, categoricals=cats)
        assert "communityname" in results.ids.infos
    except Exception as e:
        raise ValueError("Identifier 'communityname' was not detected") from e


def do_inspect_target(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    df = ds.load()
    try:
        inspect_target(df=df, target="target", is_classification=ds.is_classification)
    except ValueError as e:
        message = str(e)
        if "constant after dropping NaNs" not in message:
            raise e
    except Exception as e:
        raise ValueError(f"Could not inspect target for data {dsname}") from e


def do_caching(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset
    ds.inspect(load_cached=True)


@all_ds
@pytest.mark.cached
def test_caching(dataset: tuple[str, TestDataset]) -> None:
    do_caching(dataset)


@pytest.mark.regen
@fast_ds
def test_inspect_fast(dataset: tuple[str, TestDataset]) -> None:
    do_inspect(dataset)


@pytest.mark.regen
@med_ds
def test_inspect_medium(dataset: tuple[str, TestDataset]) -> None:
    do_inspect(dataset)


@pytest.mark.regen
@slow_ds
def test_inspect_slow(dataset: tuple[str, TestDataset]) -> None:
    do_inspect(dataset)


@fast_ds
def test_inspect_targ_fast(dataset: tuple[str, TestDataset]) -> None:
    do_inspect_target(dataset)


@med_ds
def test_inspect_targ_medium(dataset: tuple[str, TestDataset]) -> None:
    do_inspect_target(dataset)


@slow_ds
def test_inspect_targ_slow(dataset: tuple[str, TestDataset]) -> None:
    do_inspect_target(dataset)


# @fast_ds
# @pytest.mark.wip
# def test_timestamp_detect_fast(dataset: tuple[str, TestDataset]) -> None:
#     do_timestamp_detection(dataset)


# @med_ds
# @pytest.mark.wip
# def test_timestamp_detect_med(dataset: tuple[str, TestDataset]) -> None:
#     do_timestamp_detection(dataset)


# @slow_ds
# @pytest.mark.wip
# def test_timestamp_detect_slow(dataset: tuple[str, TestDataset]) -> None:
#     do_timestamp_detection(dataset)


#########


# @fast_ds
# @pytest.mark.wip
# def test_detect_floats_fast(dataset: tuple[str, TestDataset]) -> None:
#     do_detect_floats(dataset)


# @med_ds
# @pytest.mark.wip
# def test_detect_floats_med(dataset: tuple[str, TestDataset]) -> None:
#     do_detect_floats(dataset)


# @slow_ds
# @pytest.mark.wip
# def test_detect_floats_slow(dataset: tuple[str, TestDataset]) -> None:
#     do_detect_floats(dataset)


# @fast_ds
# @pytest.mark.wip
# def test_detect_ids_fast(dataset: tuple[str, TestDataset]) -> None:
#     do_detect_ids(dataset)


# @med_ds
# @pytest.mark.wip
# def test_detect_ids_med(dataset: tuple[str, TestDataset]) -> None:
#     do_detect_ids(dataset)


# @slow_ds
# @pytest.mark.wip
# def test_detect_ids_slow(dataset: tuple[str, TestDataset]) -> None:
#     do_detect_ids(dataset)


@pytest.mark.fast
def test_detect_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(50)], size=500, replace=True),
        columns=["ints"],
    ).astype(int)
    str_cols = ["ints"]
    ords = inspect_str_columns(df, str_cols=str_cols)[1]
    assert "ints" in ords.infos


@pytest.mark.fast
def test_detect_probably_ordinal() -> None:
    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(30000)], size=200, replace=False),
        columns=["ints"],
    ).astype(int)
    ords, ids = inspect_str_columns(df, str_cols=["ints"])[1:3]
    assert "ints" in ids.infos
    assert "All values including possible NaNs" in ids.infos["ints"].reason


@pytest.mark.fast
def test_detect_heuristically_ordinal() -> None:
    rng = np.random.default_rng(68)
    df = DataFrame(
        data=rng.choice([*range(7)], size=10, replace=True),
        columns=["ints"],
    ).astype(int)
    ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords.infos
    assert "Integers not starting at" in ords.infos["ints"].reason

    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(8)], size=10, replace=True),
        columns=["ints"],
    ).astype(int)
    ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords.infos
    assert (
        "80% or more of unique integer values differ only by 1"
        in ords.infos["ints"].reason
    )

    rng = np.random.default_rng(68)
    df = DataFrame(
        data=rng.choice([*range(101)], size=100, replace=True),
        columns=["ints"],
    ).astype(int)
    ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords.infos
    assert "common scale max" in ords.infos["ints"].reason

    rng = np.random.default_rng(69)
    df = DataFrame(
        data=rng.choice([*range(101)], size=100, replace=True),
        columns=["ints"],
    ).astype(int)
    ords = inspect_str_columns(df, str_cols=["ints"])[1]
    assert "ints" in ords.infos
    assert "common 0-indexed scale max" in ords.infos["ints"].reason


@pytest.mark.fast
def test_detect_simple_ids() -> None:
    df = DataFrame(
        data=np.random.choice([*range(1000)], size=100, replace=False),
        columns=["ints"],
    )
    str_cols = ["ints"]
    ids = inspect_str_columns(df, str_cols=str_cols)[2]
    assert "ints" in ids.infos


if __name__ == "__main__":
    """
    Updated:

    "elder: {'timestamp': '100% of data parses as datetime'}"
    "soybean: {'date': ' 99.80% of data appears parseable as datetime data'}"
    "dgf_96f4164d-956d-4c1c-b161-68724eb0ccdc: {'date_diagnostic': '100% of data parses as datetime'}"

    """
    times = []
    names = set()
    w = get_terminal_size((81, 24))[0]
    # for dsname, ds in TIME_DATASETS.items():
    for dsname, ds in TEST_DATASETS.items():
        # TODO: for ozone_level handle funky NaNs as float
        # if dsname != "ozone_level":
        #     continue
        df = ds.load()

        print("#" * w, file=stderr)
        print(f"Checking {dsname}", file=stderr)
        try:
            start = perf_counter()
            results = inspect_data(
                df=df, target="target", grouper=None, categoricals=ds.categoricals
            )
            duration = perf_counter() - start
            times.append((dsname, duration))
        except TypeError as e:
            if "cardinality" in str(e) and dsname == "community_crime":
                continue
            else:
                raise e
        print("#" * w, file=stderr)

        # input("Continue?")
    times = sorted(times, key=lambda pair: pair[1], reverse=True)
    for dsname, ts in times:
        pprint(f"{dsname}: {ts}", indent=2, width=w)

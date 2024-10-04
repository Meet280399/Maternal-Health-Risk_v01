from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.utils.validation import check_X_y
from tqdm import tqdm

from df_analyze.testing.datasets import (
    FAST_INSPECTION,
    TestDataset,
    fast_ds,
    med_ds,
    slow_ds,
)


def do_split_cached(dataset: tuple[str, TestDataset]) -> None:
    dsname, ds = dataset

    try:
        ds.inspect(load_cached=True)
        prep = ds.prepared(load_cached=True)
    except ValueError as e:
        if dsname == "credit-approval_reproduced":
            message = str(e)
            assert "is constant" in message
        else:
            raise e
    except Exception as e:
        raise ValueError(f"Could not prepare data: {dsname}") from e

@fast_ds
@pytest.mark.cached
def test_prep_cached_fast(dataset: tuple[str, TestDataset]) -> None:
    do_split_cached(dataset)


@med_ds
@pytest.mark.cached
def test_prep_cached_med(dataset: tuple[str, TestDataset]) -> None:
    do_split_cached(dataset)


@slow_ds
@pytest.mark.cached
def test_prep_cached_slow(dataset: tuple[str, TestDataset]) -> None:
    do_split_cached(dataset)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"
TESTDATA = DATADIR / "testing"

# Testing
SIMPLE_XLSX = DATADIR / "spreadsheet.xlsx"
SIMPLE_CSV = DATADIR / "spreadsheet.csv"
SIMPLE_CSV2 = DATADIR / "spreadsheet2.csv"
SIMPLE_ODS = DATADIR / "spreadsheet.ods"
COMPLEX_XLSX = DATADIR / "spreadsheet_complex.xlsx"
COMPLEX_XLSX2 = DATADIR / "spreadsheet_complex2.xlsx"
MUSHROOM_DATA = TESTDATA / "classification/mushrooms/mushrooms.parquet"
MUSHROOM_TYPES = TESTDATA / "classification/mushrooms/types.csv"
ELDER_DATA = TESTDATA / "classification/elder/measurements.csv"
ELDER_TYPES = TESTDATA / "classification/elder/types.csv"

DEFAULT_OUTDIR = Path.home().resolve() / "df-analyze-outputs"

DATAFILE = DATADIR / "MCICFreeSurfer.mat"
DATA_JSON = DATAFILE.parent / "mcic.json"
CLEAN_JSON = DATAFILE.parent / "mcic_clean.json"
UNCORRELATED = DATADIR / "mcic_uncorrelated_cols.json"

CLASSIFIERS = ["rf", "svm", "dtree", "mlp", "bag", "dummy", "lgb"]
REGRESSORS = ["linear", "rf", "svm", "adaboost", "gboost", "mlp", "knn", "lgb"]
DIMENSION_REDUCTION = ["pca", "kpca", "umap"]
WRAPPER_METHODS = ["step-up", "step-down"]
UNIVARIATE_FILTER_METHODS = ["d", "auc", "pearson", "t-test", "u-test", "chi", "info"]
"""
info   - information gain / mutual info
chi    - chi-squared
u-test - Mann-Whitney U

See https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
"""
MULTIVARIATE_FILTER_METHODS = ["fcbf", "mrmr", "relief"]
"""
FCBF - fast correlation-based filter\n
mRMR - minimal-redundancy-maximal-relevance\n
CMIM - conditional mutual information maximization\n
relief - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6299836/
         we likely want MultiSURF (not MultiSURF*), available in
         https://github.com/EpistasisLab/scikit-rebate

See:
https://en.wikipedia.org/wiki/Feature_selection
https://en.wikipedia.org/wiki/Relief_(feature_selection)


https://www.frontiersin.org/articles/10.3389/fbinf.2022.927312/full
Another popular family of filter algorithms is the Relief-based algorithm (RBA)
family (e.g., Relief (Kira and Rendell, 1992), ReliefF (Kononenko, 1994), TURF
(Moore and White, 2007), SURF (Greene et al., 2009), SURF* (Greene et al.,
2010), MultiSURF (Urbanowicz et al., 2018a), MultiSURF* (Granizo-Mackenzie and
Moore, 2013), etc…). Relief does not exhaustively search for feature
interactions. Instead, it scores the importance of a feature according to how
well the feature’s value distinguishes samples that are similar to each other
(e.g., similar genotype) but belong to different classes (e.g., case and
control). Notably, RBAs can detect pair-wise feature interactions, some RBAs
(e.g., ReliefF, MultiSURF) can even detect higher order (>2 way) interactions
(Urbanowicz et al., 2018a).
"""
FEATURE_SELECTIONS = ["step-up", "step-down", "pca", "kpca", "d", "auc", "pearson", "none"]
FEATURE_CLEANINGS = ["correlated", "constant", "lowinfo"]
CLEANINGS_SHORT = ["corr", "const", "info"]
HTUNE_VAL_METHODS = ["holdout", "kfold", "k-fold", "loocv", "mc", "none"]


class __Sentinel:
    pass


SENTINEL = __Sentinel()

VAL_SIZE = 0.20
SEED = 69

MAX_STEPWISE_SELECTION_N_FEATURES = 200
"""Number of features to warn user about feature selection problems"""
MAX_PERF_N_FEATURES = 500
"""Number of features to warn users about general performance problems"""

N_CAT_LEVEL_MIN = 20
"""
Minimum required number of samples for level of a categorical variable to be
considered useful in 5-fold analyses.

Details
-------

For each level of categorical variable to be predictively useful, there must
be enough samples to be statistically meaningful (or to allow some reasonable
generalization) in each fold used for fitting or analyses. Roughly, this
means each fold needs to see at least 10-20 samples of each level (depending
on how strongly / cleanly the level relates to other features - ultimately
this is just a heuristic). Assuming k-fold is used for validation, then this
means about (1 - 1/k) times 10-20 samples per categorical level would a
reasonable default minimum requirement one might use for culling categorical
levels. Under the typical assumption of k=5, this means we require useful /
reliable categorical levels to have 8-16 samples each.

Inflation is to categorical variables as noise is to continuous ones.
"""

# from https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
NAN_STRINGS = [
    "",
    "-1.#IND",
    "-1.#QNAN",
    "-nan",
    "-NaN",
    "#N/A N/A",
    "#N/A",
    "#NA",
    "<NA>",
    "1.#IND",
    "1.#QNAN",
    "n/a",
    "N/A",
    "NA",
    "nan",
    "NaN",
    "Nan",
    "None",
    "null",
    "NULL",
    "-",
    "_",
]

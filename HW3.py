# %% [markdown]
# # DSCI 552 Homework 3
# ## 1. Time Series Classification Part 1: Feature Creation/Extraction
#
# The AReM files contain six sensor time series per instance.  The fixed split
# requested in the homework is recorded below for use in Homework 4, but this
# notebook only performs the Homework 3 feature work.

# %%
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from IPython.display import display


DATA_ROOT = Path("data/AReM")
ACTIVITIES = (
    "bending1",
    "bending2",
    "cycling",
    "lying",
    "sitting",
    "standing",
    "walking",
)
SIGNALS = (
    "avg_rss12",
    "var_rss12",
    "avg_rss13",
    "var_rss13",
    "avg_rss23",
    "var_rss23",
)
TEST_LAST_DATASET = {
    "bending1": 2,
    "bending2": 2,
    "cycling": 3,
    "lying": 3,
    "sitting": 3,
    "standing": 3,
    "walking": 3,
}
EXPECTED_TIMES = pd.Index(np.arange(0, 120_000, 250), name="time")


def dataset_number(path: Path) -> int:
    """Return the dataset number encoded in a file name such as dataset12.csv."""
    match = re.fullmatch(r"dataset(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected AReM file name: {path.name}")
    return int(match.group(1))


def read_instance(path: Path) -> pd.DataFrame:
    """Read either comma- or whitespace-delimited AReM data after its comments."""
    rows = [
        re.split(r"[\s,]+", line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    frame = pd.DataFrame(rows).replace("", np.nan).dropna(axis=1, how="all")

    if frame.shape[1] != 7:
        raise ValueError(f"{path} has {frame.shape[1]} columns; expected 7.")

    frame.columns = ("time", *SIGNALS)
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except ValueError as error:
        raise ValueError(f"Non-numeric value in {path}") from error
    if frame["time"].duplicated().any() or not frame["time"].isin(EXPECTED_TIMES).all():
        raise ValueError(f"Unexpected timestamps in {path}")

    frame = frame.set_index("time").reindex(EXPECTED_TIMES)
    frame[list(SIGNALS)] = frame[list(SIGNALS)].interpolate(
        method="index", limit_area="inside"
    )
    if frame[list(SIGNALS)].isna().any().any():
        raise ValueError(f"Missing value in {path}")
    return frame.reset_index()


# %% [markdown]
# ### 1(c)(i) Common time-domain features
#
# Common groups are location (mean and median), extrema and range, dispersion
# (variance, standard deviation, IQR), distribution shape (skewness and
# kurtosis), energy (for example RMS), and temporal behavior (for example
# zero-crossing rate and autocorrelation).  For this homework, the required
# features are minimum, maximum, mean, median, standard deviation, first
# quartile, and third quartile.

# %% [markdown]
# ### 1(b) and 1(c)(ii) Fixed split and feature table
#
# The parser accepts both comma- and whitespace-delimited files because a few
# AReM files have minor formatting differences.  The one missing interior time
# point is restored by linear interpolation so every instance has 480 values.
# Standard deviations use `ddof=1` (the sample standard deviation), and
# quartiles use pandas' default linear interpolation.

# %%
records: list[dict[str, float | int | str]] = []

for activity in ACTIVITIES:
    files = sorted((DATA_ROOT / activity).glob("dataset*.csv"), key=dataset_number)
    if not files:
        raise FileNotFoundError(f"No AReM datasets found for {activity}.")

    for path in files:
        file_number = dataset_number(path)
        time_series = read_instance(path)
        record: dict[str, float | int | str] = {
            "instance": len(records) + 1,
            "activity": activity,
            "dataset": file_number,
            "split": "test" if file_number <= TEST_LAST_DATASET[activity] else "train",
        }
        for signal in SIGNALS:
            values = time_series[signal]
            record[f"{signal}__min"] = values.min()
            record[f"{signal}__max"] = values.max()
            record[f"{signal}__mean"] = values.mean()
            record[f"{signal}__median"] = values.median()
            record[f"{signal}__std"] = values.std(ddof=1)
            record[f"{signal}__q1"] = values.quantile(0.25)
            record[f"{signal}__q3"] = values.quantile(0.75)
        records.append(record)

feature_table = pd.DataFrame(records)
feature_columns = [
    f"{signal}__{statistic}"
    for signal in SIGNALS
    for statistic in ("min", "max", "mean", "median", "std", "q1", "q3")
]

assert feature_table.shape == (88, 46)
assert len(feature_columns) == 42
assert feature_table[feature_columns].notna().all().all()
assert (feature_table["split"] == "test").sum() == 19
assert (feature_table["split"] == "train").sum() == 69

split_counts = (
    feature_table.groupby(["activity", "split"], sort=False)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["train", "test"], fill_value=0)
)

display(split_counts)
display(feature_table.head())

# %% [markdown]
# ### 1(c)(iii) 90% bootstrap confidence intervals for feature standard deviations
#
# Each interval resamples the 88 instance-level values of one extracted feature.
# The reported interval is the 5th to 95th percentile of 1,000 bootstrap sample
# standard deviations.

# %%
N_BOOTSTRAP = 1_000
rng = np.random.default_rng(552)


def bootstrap_std_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Return the sample standard deviation and its 90% percentile bootstrap CI."""
    values = np.asarray(values, dtype=float)
    sample_std = float(np.std(values, ddof=1))
    resample_indices = rng.integers(0, values.size, size=(N_BOOTSTRAP, values.size))
    bootstrap_stds = np.std(values[resample_indices], axis=1, ddof=1)
    ci_low, ci_high = np.quantile(bootstrap_stds, [0.05, 0.95])
    return sample_std, float(ci_low), float(ci_high)


ci_rows = []
for feature in feature_columns:
    sample_std, ci_low, ci_high = bootstrap_std_ci(feature_table[feature].to_numpy())
    ci_rows.append(
        {
            "feature": feature,
            "sample_std": sample_std,
            "ci_low_90": ci_low,
            "ci_high_90": ci_high,
        }
    )

confidence_intervals = pd.DataFrame(ci_rows)
assert confidence_intervals.shape == (42, 4)
assert np.isfinite(confidence_intervals[["sample_std", "ci_low_90", "ci_high_90"]]).all().all()
display(confidence_intervals)

# %% [markdown]
# ### 1(c)(iv) Three important features
#
# I select **minimum, mean, and maximum**.  They respectively capture the low
# end, typical level, and high end of a sensor signal.  Together they preserve
# each series' basic level and range without using three nearly redundant
# measures of center or spread.

# %% [markdown]
# ## 2. ISLR 3.7.4
#
# **(a)** The cubic model has training RSS less than or equal to the linear
# model's training RSS because the linear model is nested inside it.
#
# **(b)** If the true relationship is linear, the linear model should have lower
# test RSS on average.  The extra cubic terms add variance without reducing
# bias.
#
# **(c)** The cubic model again has training RSS less than or equal to the
# linear model's training RSS.
#
# **(d)** There is not enough information to compare test RSS.  It depends on
# how nonlinear the true relationship is relative to the extra variance of the
# cubic fit.

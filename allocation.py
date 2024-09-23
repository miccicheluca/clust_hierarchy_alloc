from tqdm import tqdm

from typing import Optional, Dict, Tuple

import polars as pl

import datetime as dt

import weight_schemes as ws


class PortfolioAllocation:
    def __init__(
        self,
        df_ptf: pl.DataFrame,
        weight_recipe: Dict,
        start_alloc: Optional[dt.datetime] = None,
        end_alloc: Optional[dt.datetime] = None,
        subsample: Optional[str] = None,
        sample_time: Optional[str] = None,
        rebal_frequency: int = 1,
    ):
        self.allocation_history = None

        self.df_ptf = df_ptf

        self.weight_recipe = weight_recipe

        if start_alloc is not None and end_alloc is not None:
            self.df_ptf = self.df_ptf.filter(pl.col("date").is_between(start_alloc, end_alloc))

        self.subsample = subsample
        self.sample_time = sample_time
        self.rebal_frequency = rebal_frequency

        # subsample

        if (self.subsample is not None) & (self.sample_time is not None):
            self._subsample()

        self.dates = sorted(self.df_ptf["date"].unique())

        # compute relative weights & volatility target constraints

        self._compute_relative_weights()

    def _subsample(self) -> None:
        """

        Subsample dataframe to weekly/monthly (last or first day).

        """

        df = (
            self.df_ptf.with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                    pl.col("date").dt.week().alias("week"),
                    pl.col("date").dt.day().alias("day"),
                ]
            )
            .with_columns([(pl.col(self.subsample) % self.rebal_frequency).alias("rebal")])
            .filter(pl.col("rebal") == 0)
        )

        if self.subsample in ["week", "month"]:
            if self.sample_time == "first":
                dates = df.group_by(["year", self.subsample]).first().select("date").to_series()

            elif self.sample_time == "last":
                dates = df.group_by(["year", self.subsample]).last().select("date").to_series()

            else:
                raise ValueError("sample_time should be first or last")

        elif self.subsample == "day":
            dates = df.select("date").to_series().unique()

        else:
            raise ValueError("subsample should be week, month or day")

        self.df_ptf = self.df_ptf.filter(pl.col("date").is_in(dates))

    def _compute_relative_weights(self) -> None:
        """Computes weight between (0 and 1) for every asset. Sum is 1."""

        store = []

        for k, v in self.weight_recipe.items():
            func = v[0]

            args = v[1]

            out = self.df_ptf.filter(pl.col("alloc_ptf") == k).pipe(func, **args)

            store.append(out)

        self.df_ptf = pl.concat(store)

import sys
import polars as pl
import numpy as np
import pandas as pd
import datetime as dt

from typing import List, Optional
from tqdm import tqdm

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    optimal_leaf_ordering,
    leaves_list,
)
from scipy.spatial.distance import squareform

import top_down as td

from allocation_dataclass import (
    hc_args,
    hc_sim_matrix_args,
    hc_risk_args,
    hc_rets_args,
    current_date_infos,
)

sys.path.append("../utils")
from utils import ColumnName


class HRP:
    def __init__(self, df_ref: pl.DataFrame, col_names: ColumnName = ColumnName()):
        """
        df_ref: raw data set

        The function creates here a TxN matrix of stock returns which will be used to compute correlation and covariance matrix in the process.
        T is the number of date, N is the number of stock in the universe.
        """
        self.col_names = col_names
        self.df_reference = self._pivot_data(df=df_ref)

    def _pivot_data(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pivot(
            index=self.col_names.date, on=self.col_names.asset_id, values=self.col_names.rets
        )

    def _pariwise_exp_cov(self, X: pl.Series, Y: pl.Series, alpha: float) -> float:
        """Compute alpha exponentially weighted covariance matrix between to series X and Y"""
        covariation = (X - X.mean()) * (Y - Y.mean())

        return covariation.ewm_mean(alpha=alpha)[-1]

    def _compute_sim_matrix(
        self,
        infos: current_date_infos,
        args_sim_matrix: hc_sim_matrix_args,
    ) -> pd.DataFrame:
        """ "
        infos: corresponds to the actual rebal infos (date, assets in portfolio, )
        args_corr: corresponds to the correlation matrix computation infos (lag, type and method)

        Compute the correlation matrix which will be used to create the hierarchi through our univers
        """
        df_res = None

        if args_sim_matrix.method == "correlation":
            df_res = pl.read_parquet(
                f"C:/Users/HP/Desktop/Work/Projects/factor_investing/data/prep_data/corr/corr_factor_date_{infos.rebal_date}_window_{args_sim_matrix.window}"
            ).to_pandas()

        return df_res

    def cov2corr(self, df_cov: pd.DataFrame) -> pd.DataFrame:
        """Derive the correlation matrix from a covariance matrix"""

        std = np.sqrt(np.diag(df_cov))

        corr = np.clip(df_cov / np.outer(std, std), a_min=-1.0, a_max=1.0)

        return corr

    def _compute_cluster(self, df_dist: pd.DataFrame, method: str) -> np.ndarray:
        return linkage(df_dist, method)

    def _compute_dist(self, df_sim_matrix: pd.DataFrame, method: str) -> pd.DataFrame:
        if method == "ang_dist":
            dist = ((1 - df_sim_matrix) / 2) ** 0.5
            return squareform(dist, checks=False)

        if method == "abs_ang_dist":
            dist = ((1 - df_sim_matrix.abs()) / 2) ** 0.5
            return squareform(dist, checks=False)

        if method == "square_ang_dist":
            dist = ((1 - df_sim_matrix**2) / 2) ** 0.5
            return squareform(dist, checks=False)

        raise ValueError("Method should be in {'abs_dis', 'abs_ang_dist', 'square_ang_dist'}")

    def _compute_quasi_diag(self, arr_linkage: np.ndarray, df_dist: pd.DataFrame) -> List[int]:
        """rearrangement according to clusters. This list is used to creat the quasi-diagonal matrix"""
        return leaves_list(optimal_leaf_ordering(arr_linkage, df_dist))

    def _compute_weight(
        self,
        method: str,
        arr_linkage: np.array,
        rebal_infos: current_date_infos,
        df_sort_stocks_id: pl.DataFrame,
        args_hc: Optional[hc_args] = None,
        args_risk: Optional[hc_risk_args] = None,
        args_rets: Optional[hc_rets_args] = None,
    ):
        """
        method: top-down or bottom-up apporach (bottum up need to be implemented)
        arr_linkage: represents cluster hierarchi given by scipy
        rebal_infos: represents the current rebal infos (date, assets in ptf etc..)
        df_sort_stock_id: corresponds to the identification of stocks and their id in the hierarchi
        args_hc: infos about the computation of the hierarchi
        args_cov: infos about the computation of covariance matrix for top-down approach
        args_rets: infos about the computation of returns if you want to use HARP approach
        args_vol_pred: infos if you want to use a "future aversion" method

        if you want to use equaly weighted hrp: give only args_hc
        if you want to use simple hrp: give args_hc & args_cov
        if you want to use harp: give args_hc, args_cov, args_rets
        """
        weights = pl.DataFrame

        if method == "top_down":
            weights = td.compute_top_down_w(
                arr_linkage=arr_linkage,
                rebal_infos=rebal_infos,
                df_sort_stocks_id=df_sort_stocks_id,
                args_hc=args_hc,
                args_risk=args_risk,
                args_rets=args_rets,
            )

        return weights

    def _compute_df_ref_temp(
        self,
        rebal_assets: List[str],
    ) -> pl.DataFrame:
        return self.df_reference.select(pl.col([self.col_names.date] + rebal_assets))

    def _compute_current_date_infos(
        self, df_prediction: pl.DataFrame, rebal_date: dt.datetime, args_hc: hc_args
    ) -> current_date_infos:
        df_pred_temp = df_prediction.filter(
            pl.col(self.col_names.date) == rebal_date, pl.col("date") > pl.col("available_date")
        )

        rebal_assets = (
            df_pred_temp.select(self.col_names.asset_id).to_series().unique().sort().to_list()
        )

        df_ref_temp = self._compute_df_ref_temp(
            rebal_assets=rebal_assets,
        )

        if args_hc.dyn_weighting:
            args_hc.weight_max = (1 / len(rebal_assets)) * 5

        return (
            current_date_infos(
                df_reference=df_ref_temp,
                rebal_date=rebal_date,
                rebal_assets=rebal_assets,
                df_pred=df_pred_temp,
            ),
            args_hc,
        )

    def compute_hrp(
        self,
        df_ptf: pl.DataFrame,
        args_hc: hc_args,
        args_sim_matrix: hc_sim_matrix_args,
        args_risk: Optional[hc_risk_args] = None,
        args_rets: Optional[hc_rets_args] = None,
    ) -> pl.DataFrame:
        out = []

        dates = df_ptf.select(self.col_names.date).to_series().unique().sort()

        for d in tqdm(dates):
            rebal_infos, args_hc = self._compute_current_date_infos(
                df_prediction=df_ptf, rebal_date=d, args_hc=args_hc
            )

            # STEP 1: Compute distance matrix from similarity matrix (here correlation matrix)
            df_sim_matrix = self._compute_sim_matrix(
                infos=rebal_infos, args_sim_matrix=args_sim_matrix
            )

            df_dist = self._compute_dist(
                df_sim_matrix=df_sim_matrix, method=args_hc.distance_method
            )

            # STEP 2: Compute clusters
            arr_linkage = self._compute_cluster(df_dist=df_dist, method=args_hc.linkage_method)

            sort_stocks_id = self._compute_quasi_diag(arr_linkage, df_dist).astype(np.int64)

            assets_id = pl.DataFrame(
                {
                    "id": list(range(df_sim_matrix.shape[0])),
                    self.col_names.asset_id: df_sim_matrix.columns.to_list(),
                }
            )

            df_sort_stock_id = pl.DataFrame({"id": sort_stocks_id}).join(
                assets_id, on="id", how="left"
            )

            # STEP 3-4: Compute rebal weights thanks to the hierarchi
            weights = self._compute_weight(
                method=args_hc.weights_method,
                arr_linkage=arr_linkage,
                rebal_infos=rebal_infos,
                df_sort_stocks_id=df_sort_stock_id,
                args_hc=args_hc,
                args_risk=args_risk,
                args_rets=args_rets,
            )

            assets_id = assets_id.join(weights, on="id", how="left")

            out.append(
                rebal_infos.df_pred.join(
                    assets_id.select([self.col_names.asset_id, "relative_weight"]),
                    on=[self.col_names.asset_id],
                    how="left",
                )
            )

        return pl.concat(out)

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from bubble_data_collector import load_bubble_config, get_bubble_config


class Preprocessor:
    """
    Transform financial time-series data to identify bubble boundaries and 
    generate firm-level features and survival metrics.
    """

    def __init__(self, drop_threshold: float = 0.3, _lambda: float = 1):
        """
        Initialize preprocessing parameters and load bubble configuration.

        Parameters
        ----------
        drop_threshold : float
            Hyperparameter for price drop to identify bubble end.
        _lambda : float
            Hyperparameter for breadth adjustment in time series construction.
        """
        self.drop_threshold = 0.3
        self._lambda = 1
        self.bubble_config = load_bubble_config()

        assert 0 < self.drop_threshold < 1, "drop_threshold must be in (0, 1)"
        assert self._lambda >= 0, "_lambda must be non-negative"


    def transform(self, bubble_type: str, is_train: bool = True) -> pd.DataFrame:
        """
        Load bubble data, compute bubble window, and aggregate firm-level features.

        Parameters
        ----------
        bubble_type : str
            Bubble identifier (e.g., 'dotcom', 'crypto', 'ev').
        is_train : bool
            Whether to compute and attach survival probability labels.

        Returns
        -------
        pd.DataFrame
            Aggregated firm-level dataset for modeling.
        """
        assert isinstance(bubble_type, str) and bubble_type, "bubble_type must be a non-empty string"
        assert isinstance(is_train, bool), "is_train must be a boolean"

        bubble_params = get_bubble_config(bubble_type, self.bubble_config)
        assert "data_path" in bubble_params and "start_date" in bubble_params

        df = pd.read_csv(bubble_params['data_path'])
        assert not df.empty, "Loaded dataset is empty"

        df = df[df['date'] >= bubble_params['start_date']]
        df["date"] = pd.to_datetime(df["date"])
        df["quarter"] = df["date"].dt.to_period("Q")

        y = self.create_time_series(df, 'quarter', 'prc')
        assert not y.empty, "Constructed time series is empty"

        if is_train:
            t_start, t_end = self.identify_bubble_bounds(y)
        else:
            t_start, t_end = df["quarter"].min(), df["quarter"].max()

        print(f"===========:{bubble_type} Bubble window: ===========")
        print(f"Start → {pd.to_datetime(bubble_params['start_date']).to_period("Q")}")
        print(f"Peak → {t_start}")
        print(f"End  → {t_end}")

        survival_prob = self.calculate_survival_prob(t_start, t_end, df)

        df_processed = df[df['quarter'].between(t_start, t_end)].copy()
        assert not df_processed.empty, "No data found within bubble window"

        df_processed = (
            df_processed
            .groupby('permno')
            .mean(numeric_only=True)
            .reset_index()
        )


        if is_train:
            df_processed['survival_prob'] = df_processed['permno'].map(survival_prob)
            df_processed = df_processed.dropna(subset=["survival_prob"])

        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

        return df_processed


    def identify_bubble_bounds(self, y: pd.Series):
        """
        Identify bubble peak and end based on rolling drawdown.

        Parameters
        ----------
        y : pd.Series
            Time series used for bubble detection.

        Returns
        -------
        tuple
            (t_start, t_end) representing peak and end quarters.
        """
        assert isinstance(y, pd.Series), "y must be a pandas Series"
        assert not y.empty, "Input time series is empty"

        rolling_peak = y.rolling(6, min_periods=1).max()
        is_peak = y == rolling_peak
        drawdown = (y - rolling_peak) / rolling_peak

        hits = drawdown[drawdown <= -self.drop_threshold]
        assert not hits.empty, "No drawdown threshold crossing found"

        t_end = hits.idxmin()

        peaks = is_peak.loc[:t_end]
        assert peaks.any(), "No rolling peak found before bubble end"

        t_start = peaks[peaks].index[-1]

        return t_start, t_end


    def create_time_series(self, df, date_col, value_col="prc"):
        """
        Construct a breadth-adjusted log-price time series.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        date_col : str
            Time grouping column.
        value_col : str
            Price column name.

        Returns
        -------
        pd.Series
            Time-indexed series for bubble detection.
        """
        assert isinstance(df, pd.DataFrame), "df must be a DataFrame"
        assert date_col in df.columns, f"{date_col} not in DataFrame"
        assert value_col in df.columns, f"{value_col} not in DataFrame"
        assert "permno" in df.columns, "permno column missing"

        grouped = df.groupby(date_col)
        mean_price = grouped[value_col].mean()
        assert not mean_price.empty, "Mean price series is empty"

        n_firms = grouped["permno"].nunique()
        breadth_factor = n_firms / n_firms.max()

        y = (np.log(mean_price) + self._lambda * np.log(breadth_factor)).sort_index()
        return y


    def calculate_survival_prob(self, t_start, t_end, df):
        """
        Compute normalized survival probability based on price drawdown.

        Parameters
        ----------
        t_start : period
            Bubble peak quarter.
        t_end : period
            Bubble end quarter.
        df : pd.DataFrame
            Full dataset.

        Returns
        -------
        pd.Series
            Normalized survival probability per firm.
        """
        assert isinstance(df, pd.DataFrame), "df must be a DataFrame"
        assert "quarter" in df.columns, "quarter column missing"
        assert "permno" in df.columns, "permno column missing"
        assert "prc" in df.columns, "prc column missing"

        value_start = (
            df.loc[df['quarter'] == t_start]
            .groupby('permno')['prc']
            .mean()
        )
        value_end = (
            df.loc[df['quarter'] == t_end]
            .groupby('permno')['prc']
            .mean()
        )

        drop_percentage = (value_start - value_end) / value_start
        drop_percentage = drop_percentage.dropna()
        assert not drop_percentage.empty, "Drop percentage computation failed"

        survival_percentage = drop_percentage * -1
        survival_prob = (
            (survival_percentage - survival_percentage.min()) /
            (survival_percentage.max() - survival_percentage.min())
        )

        assert survival_prob.between(0, 1).all(), \
            "Survival Probability percentages not in [0, 1]"

        return survival_prob
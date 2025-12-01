import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from bubble_data_collector import load_bubble_config, get_bubble_config

class Preprocessor:
    def __init__(self):
        self.drop_threshold = 0.3
        self._lambda = 1
        self.bubble_config = load_bubble_config()

    def transform(self, bubble_type: str, is_train: bool = True) -> pd.DataFrame:

        bubble_params = get_bubble_config(bubble_type, self.bubble_config)
        df = pd.read_csv(bubble_params['data_path'])
        
        df = df[df['date'] >= bubble_params['start_date']]
        df["date"] = pd.to_datetime(df["date"])  
        df["quarter"] = df["date"].dt.to_period("Q")

        y = self.create_time_series(df, 'quarter', 'prc')
        if is_train:
            t_start, t_end = self.identify_bubble_bounds(y)
        else:
            t_start, t_end = df["quarter"].min(), df["quarter"].max()

        print("===========: Bubble window: ===========")
        print(f"Peak → {t_start}")
        print(f"End  → {t_end}")

        survival_prob = self.calculate_survival_prob(t_start, t_end, df)

        df_processed = df[df['quarter'].between(t_start, t_end)].copy()
        df_processed = df_processed.groupby('permno').mean(numeric_only=True).reset_index()
       
        if is_train:
            df_processed['survival_prob'] = df_processed['permno'].map(survival_prob)
            df_processed = df_processed.dropna(subset=["survival_prob"])
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

        return df_processed

        
    def identify_bubble_bounds(self, y: pd.Series):
        """
        Identify the end of the bubble based on drawdown from peak.
        Returns
        -------
        <Year><Quarter> where bubble ends, or None if not found.
        """

        assert isinstance(y, pd.Series)

        rolling_peak = y.rolling(6, min_periods=1).max()
        is_peak = y == rolling_peak
        drawdown = (y - rolling_peak) / rolling_peak

        hits = drawdown[drawdown <= -self.drop_threshold]
        t_end = hits.idxmin()

        peaks = is_peak.loc[:t_end]
        assert peaks.any(), "No rolling peak found before bubble end"

        t_start = peaks[peaks].index[-1]

        return t_start, t_end

    def create_time_series(self, df, date_col, value_col="prc"):
        grouped = df.groupby(date_col)
        mean_price =  grouped[value_col].mean()

        n_firms = grouped["permno"].nunique()
        breadth_factor = n_firms / n_firms.max()

        y = (np.log(mean_price) + self._lambda * np.log(breadth_factor)).sort_index()
        return y



    def calculate_survival_prob(self, t_start, t_end, df):
        value_start = df.loc[df['quarter'] == t_start].groupby('permno')['prc'].mean()
        value_end = df.loc[df['quarter'] == t_end].groupby('permno')['prc'].mean()

        drop_percentage = (value_start - value_end) / value_start
        drop_percentage = drop_percentage.dropna()
        survival_percentage = drop_percentage * -1

        survival_prob = ((survival_percentage - survival_percentage.min()) / (survival_percentage.max() - survival_percentage.min()))
        assert survival_prob.between(0, 1).all(), "Survival Probability percentages not in [0, 1]"
        return survival_prob
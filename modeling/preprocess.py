import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.drop_threshold = 0.3

    def transform(self, df: pd.DataFrame):

        df["date"] = pd.to_datetime(df["date"])
        df["quarter"] = df["date"].dt.to_period("Q")

        y = self.create_time_series(df, 'quarter', 'prc')
        t_start, t_end = self.identify_bubble_bounds(y)

        print("===========: Bubble window: ===========")
        print(f"Peak → {t_start}")
        print(f"End  → {t_end}")

        drop_percentage = self.calculate_drop_percentage(t_start, t_end, df)

        df_processed = df[df['quarter'].between(t_start, t_end)].copy()
        df_processed = df_processed.groupby('permno').mean(numeric_only=True).reset_index()
        df_processed['drop_percentage'] = df_processed['permno'].map(drop_percentage)

        print(df_processed.head())

        
    def identify_bubble_bounds(self, y: pd.Series):
        """
        Identify the end of the bubble based on drawdown from peak.
        Returns
        -------
        <Year><Quarter> where bubble ends, or None if not found.
        """

        assert isinstance(y, pd.Series)

        rolling_peak = y.rolling(12, min_periods=1).max()
        is_peak = y == rolling_peak
        drawdown = (y - rolling_peak) / rolling_peak

        hits = drawdown[drawdown <= -self.drop_threshold]
        t_end = hits.idxmin()

        peaks = is_peak.loc[:t_end]
        assert peaks.any(), "No rolling peak found before bubble end"

        t_start = peaks[peaks].index[-1]

        return t_start, t_end

    def create_time_series(self, df, date_col, value_col="prc"):
        return df.groupby(date_col)[value_col].mean().sort_index()

    def calculate_drop_percentage(self, t_start, t_end, df):
        value_start = df.loc[df['quarter'] == t_start].groupby('permno')['prc'].mean()
        value_end = df.loc[df['quarter'] == t_end].groupby('permno')['prc'].mean()

        drop_percentage = (value_start - value_end) / value_start
        drop_percentage = drop_percentage.dropna()

        drop_percentage_norm = ((drop_percentage - drop_percentage.min()) / (drop_percentage.max() - drop_percentage.min()))
        assert drop_percentage_norm.between(0, 1).all(), "Normalized drop percentages not in [0, 1]"
        return drop_percentage_norm

x = Preprocessor()
df = pd.read_csv('data/crypto_merged.csv')
x.transform(df)

df = pd.read_csv('data/ev_merged.csv')
x.transform(df)

df = pd.read_csv('data/dotcom_merged.csv')
x.transform(df)
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class RandomForestRegressorModel:
    """
    Wrapper class around sklearn's RandomForestRegressor for
    training, predicting, and evaluating a regression model
    with real-valued targets.

    This class enforces:
    - Explicit feature column selection
    - Numeric target variable
    - Training-before-prediction guarantee
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str,
        rf_params: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the Random Forest regression trainer.

        Parameters
        ----------
        feature_cols : List[str]
            Names of dataframe columns used as input features (X variables).
            Example: ["X1", "X2", "X3"]

        target_col : str
            Name of the dataframe column used as the regression target (Y).
            Must be real-valued.

        rf_params : dict, optional
            Dictionary of hyperparameters passed directly to
            sklearn.ensemble.RandomForestRegressor.
            Example:
            {
                "n_estimators": 300,
                "max_depth": 20,
                "min_samples_leaf": 5
            }

        random_state : int
            Random seed for reproducibility of results.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.random_state = random_state
        self.rf_params = rf_params or {}
        self.model: Optional[RandomForestRegressor] = None

    # ------------------------------------------------------------------ #
    # Internal validation utilities
    # ------------------------------------------------------------------ #
    def _validate_dataframe(self, df: pd.DataFrame):
        """
        Validate that the dataframe contains required features
        and that the target variable is suitable for regression.

        This method is intentionally strict to avoid silent failures
        or cryptic sklearn errors later in the workflow.

        Checks performed:
        - Input must be a pandas DataFrame
        - All feature columns must be present
        - Target column must exist
        - Target column must be numeric
        """
        assert isinstance(df, pd.DataFrame), \
            "Input must be a pandas DataFrame"

        missing_features = set(self.feature_cols) - set(df.columns)
        assert not missing_features, \
            f"Missing feature columns: {missing_features}"

        assert self.target_col in df.columns, \
            f"Target column '{self.target_col}' not found in DataFrame"

        assert np.issubdtype(df[self.target_col].dtype, np.number), \
            "Target column must be numeric for regression"

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, df: pd.DataFrame):
        """
        Train the Random Forest regressor on the provided dataframe.

        Steps:
        1. Validate dataframe structure and target type
        2. Extract feature matrix X and target vector y
        3. Initialize RandomForestRegressor with provided parameters
        4. Fit the model to the data

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset containing feature columns and target column.

        Returns
        -------
        self
            Returns the fitted instance to allow method chaining.
        """
        self._validate_dataframe(df)

        X = df[self.feature_cols].values
        y = df[self.target_col].values

        self.model = RandomForestRegressor(
            random_state=self.random_state,
            **self.rf_params
        )

        self.model.fit(X, y)

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new / unseen data.

        Preconditions:
        - Model must be trained prior to calling this method.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing feature columns only (target may be present
            but is not used).

        Returns
        -------
        np.ndarray
            Predicted real-valued outputs for each row in the dataframe.
        """
        assert self.model is not None, \
            "Model must be trained before prediction"

        self._validate_dataframe(df)

        X = df[self.feature_cols].values

        return self.model.predict(X)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the trained model on labeled data using
        standard regression metrics.

        Metrics reported:
        - RMSE (Root Mean Squared Error)
        - R² (Coefficient of Determination)

        Parameters
        ----------
        df : pd.DataFrame
            Evaluation dataset containing both features and target.

        Returns
        -------
        Dict[str, float]
            Dictionary with evaluation metrics.
            Example:
            {
                "rmse": 2.31,
                "r2": 0.87
            }
        """
        assert self.model is not None, \
            "Model must be trained before evaluation"

        y_true = df[self.target_col].values
        y_pred = self.predict(df)

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": rmse,
            "r2": r2
        }
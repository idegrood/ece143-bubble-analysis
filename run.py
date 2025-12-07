from modeling.preprocess import Preprocessor
from modeling.random_forest import RandomForestRegressorModel
from bubble_type import BUBBLE_TYPE
import pandas as pd
import numpy as np


class ModelRunner:
    """
    A utility class that orchestrates preprocessing, model training,
    and model inference for different bubble-type datasets.
    """

    def __init__(self, model):
        """
        Initialize the ModelRunner with a model and a Preprocessor.

        Parameters
        ----------
        model : RandomForestRegressorModel
            A model instance implementing .train(df) and .predict(df)
        """
        assert model is not None, "ModelRunner requires a valid model instance."
        self.preprocessor = Preprocessor()
        self.model = model

    def run_preprocessing(self, bubble_types: list, is_train: bool = True) -> pd.DataFrame:
        """
        Run preprocessing for a list of bubble types and return a concatenated DataFrame.

        Parameters
        ----------
        bubble_types : list
            List of bubble type values (strings or enums) to preprocess.
        is_train : bool, optional
            If True, applies preprocessing in training mode (fit + transform).
            If False, applies only transform using already-fitted preprocessing.

        Returns
        -------
        pd.DataFrame
            Preprocessed and concatenated DataFrame for all bubble types.
        """
        assert isinstance(bubble_types, list), "bubble_types must be a list."
        assert len(bubble_types) > 0, "bubble_types list cannot be empty."

        dfs_after = []
        for bt in bubble_types:
            assert bt is not None, "Bubble type values cannot be None."
            dfs_after.append(self.preprocessor.transform(bt, is_train))

        df_processed = pd.concat(dfs_after, ignore_index=True)
        assert isinstance(df_processed, pd.DataFrame), "Preprocessing must return a DataFrame."

        return df_processed

    def run_training(self, df: pd.DataFrame):
        """
        Train the model on a processed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns and target column.

        Returns
        -------
        Trained model object
        """
        assert isinstance(df, pd.DataFrame), "Training data must be a pandas DataFrame."
        assert not df.empty, "Training DataFrame cannot be empty."
        assert hasattr(self.model, "train"), "Model must implement a train(df) method."

        return self.model.train(df)

    def run_inference(self, df: pd.DataFrame) -> np.ndarray:
        """
        Run inference using the trained model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing only feature columns.

        Returns
        -------
        np.ndarray
            Array of predicted survival probabilities.
        """
        assert isinstance(df, pd.DataFrame), "Inference data must be a pandas DataFrame."
        assert not df.empty, "Inference DataFrame cannot be empty."
        assert hasattr(self.model, "predict"), "Model must implement a predict(df) method."

        preds = self.model.predict(df)
        assert isinstance(preds, (np.ndarray, list)), "Predictions must be ndarray or list."

        return np.array(preds)


def main():
    """
    Main execution function:
    - Defines feature and target columns
    - Preprocesses CRYPTO, DOTCOM, EV datasets for training
    - Preprocesses AI dataset for inference
    - Trains a RandomForestRegressorModel
    - Runs predictions and prints feature importances + results
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    feature_cols = ['prc', 'mktcap', 'ps', 'pb', 'pe', 'ev', 'ev_sales', 'ps_z', 'pb_z', 'ev_sales_z']
    target_col = 'survival_prob'

    rf_model = RandomForestRegressorModel(feature_cols=feature_cols, target_col=target_col)
    runner = ModelRunner(model=rf_model)

    df_processed = runner.run_preprocessing([
        BUBBLE_TYPE.CRYPTO.value,
        BUBBLE_TYPE.DOTCOM.value,
        BUBBLE_TYPE.EV.value
    ])

    df_predict = runner.run_preprocessing([BUBBLE_TYPE.AI.value], is_train=False)

    rf_model = runner.run_training(df_processed)

    predicted = runner.run_inference(df_predict[feature_cols])

    print("========= Feature Importances: =========")
    importances = rf_model.model.feature_importances_
    for name, val in zip(feature_cols, importances):
        print(f"{name}: {val:.4f}")

    df_predict['survival_prob_predicted'] = predicted

    print("================ Predictions: =====================")
    print(df_predict[['permno', 'survival_prob_predicted', 'prc', 'mktcap', 'ps', 'pb', 'pe']].head(10))


if __name__ == "__main__":
    main()

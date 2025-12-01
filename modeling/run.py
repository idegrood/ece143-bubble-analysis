from preprocess import Preprocessor
from random_forest import RandomForestRegressorModel
from bubble_type import BUBBLE_TYPE
import pandas as pd
import numpy as np

class ModelRunner:
    def __init__(self, model):
        self.preprocessor = Preprocessor()
        self.model = model
    
    def run_preprocessing(self, bubble_types: list, is_train: bool = True):
        dfs_after = []
        for bt in bubble_types:
            dfs_after.append(self.preprocessor.transform(bt, is_train))
        df_processed = pd.concat(dfs_after, ignore_index=True)
        return df_processed
    
    def run_training(self, df: pd.DataFrame):
        return self.model.train(df)
    
    def run_inference(self, df: pd.DataFrame):
        return self.model.predict(df)

def main():
    feature_cols = ['prc', 'mktcap','ps', 'pb', 'pe', 'ev', 'ev_sales', 'ps_z', 'pb_z', 'ev_sales_z']
    target_col = 'survival_prob'
    rf_model = RandomForestRegressorModel(feature_cols=feature_cols, target_col=target_col)
    runner = ModelRunner(model=rf_model)

    df_processed = runner.run_preprocessing([BUBBLE_TYPE.CRYPTO.value, BUBBLE_TYPE.DOTCOM.value, BUBBLE_TYPE.EV.value])
    df_predict = runner.run_preprocessing([BUBBLE_TYPE.AI.value], is_train=False)
    rf_model = runner.run_training(df_processed)
    predicted = runner.run_inference(df_predict[feature_cols])
   
    print("========= Feature Importances: =========")
    importances = rf_model.model.feature_importances_
    for name, val in zip(feature_cols, importances):
        print(f"{name}: {val:.4f}")

    df_predict['survival_prob_predicted'] = predicted

    print("================ Predictions: =====================")
    print(df_predict[['permno', 'survival_prob_predicted']].head(10))

if __name__ == "__main__":
    main()
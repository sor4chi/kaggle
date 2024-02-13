import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import constant as const
from models.lightgbm_classifier import model as LGBM
from models.xgboost_classifier import model as XGB
from models.catboost_classifier import model as CatBoost
from processor import preprocess, inverse_map_labels
import os
import json

def main():
    train_df = pd.read_csv(const.TRAIN_CSV_PATH, index_col='id')
    test_df = pd.read_csv(const.TEST_CSV_PATH, index_col='id')
    sample_submission_df = pd.read_csv(const.SAMPLE_SUBMISSION_CSV_PATH)

    # Preprocessing
    X, y, test = preprocess(train_df, test_df)

    # load best hyperparameters
    best_params = {}
    if os.path.exists('best_hyperparameters.json'):
        with open('best_hyperparameters.json', 'r') as f:
            best_params = json.load(f)

    # Hyperparameter optimization for LGBM
    # best_params_lgbm = LGBM.optimize(X, y)
    best_params_lgbm = best_params['LGBM']

    # Hyperparameter optimization for XGB
    # best_params_xgb = XGB.optimize(X, y)
    best_params_xgb = best_params['XGB']

    # Hyperparameter optimization for CatBoost
    # best_params_catboost = CatBoost.optimize(X, y)
    best_params_catboost = best_params['CatBoost']

    # Create LGBM Model
    config_lgbm = LGBM.Config(**best_params_lgbm)
    model_lgbm = LGBM.Model(config_lgbm)

    # Create XGB Model
    config_xgb = XGB.Config(**best_params_xgb)
    model_xgb = XGB.Model(config_xgb)

    # Create CatBoost Model
    config_catboost = CatBoost.Config(**best_params_catboost)
    model_catboost = CatBoost.Model(config_catboost)

    # Train LGBM
    model_lgbm.fit(X, y)

    # Train XGB
    model_xgb.fit(X, y)

    # Train CatBoost
    model_catboost.fit(X, y)

    # write to file
    with open('best_hyperparameters.json', 'w') as f:
        pretty = json.dumps({
            'LGBM': best_params_lgbm,
            'XGB': best_params_xgb,
            'CatBoost': best_params_catboost
        }, indent=2)
        f.write(pretty)

    # Predict with LGBM
    y_pred_lgbm = model_lgbm.predict(test)

    # Predict with XGB
    y_pred_xgb = model_xgb.predict(test)

    # Predict with CatBoost
    y_pred_catboost = model_catboost.predict(test)

    # Inverse map labels for LGBM
    y_pred_lgbm = inverse_map_labels(pd.DataFrame({const.TARGET_COL: y_pred_lgbm}))[const.TARGET_COL]

    # Inverse map labels for XGB
    y_pred_xgb = inverse_map_labels(pd.DataFrame({const.TARGET_COL: y_pred_xgb}))[const.TARGET_COL]

    # Inverse map labels for CatBoost
    y_pred_catboost = inverse_map_labels(pd.DataFrame({const.TARGET_COL: y_pred_catboost}))[const.TARGET_COL]

    # Ensemble predictions
    ensemble_predictions = [y_pred_lgbm, y_pred_xgb, y_pred_catboost]
    assert len(ensemble_predictions) >= 1
    final_predictions = []
    for i in range(len(ensemble_predictions[0])):
        row = [ensemble_predictions[j][i] for j in range(len(ensemble_predictions))]
        counts = {}
        for label in row:
            counts[label] = counts.get(label, 0) + 1
        if len(counts) > 1:
            print(i, counts)
        final_predictions.append(max(counts, key=counts.get))

    # Save submission
    sample_submission_df[const.TARGET_COL] = final_predictions
    sample_submission_df.to_csv(f'submission_{int(time.time())}.csv', index=False)

if __name__ == '__main__':
    main()

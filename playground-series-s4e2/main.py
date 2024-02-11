import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import constant as const
from models.lightgbm_classifier import model as LGBM
from models.xgboost_classifier import model as XGB
from processor import preprocess, inverse_map_labels

def main():
    train_df = pd.read_csv(const.TRAIN_CSV_PATH, index_col='id')
    test_df = pd.read_csv(const.TEST_CSV_PATH, index_col='id')
    sample_submission_df = pd.read_csv(const.SAMPLE_SUBMISSION_CSV_PATH)

    # Preprocessing
    X, y, test = preprocess(train_df, test_df)

    # Hyperparameter optimization for LGBM
    best_params_lgbm = LGBM.optimize(X, y)

    # Hyperparameter optimization for XGB
    best_params_xgb = XGB.optimize(X, y)


    # Create LGBM Model
    config_lgbm = LGBM.Config(**best_params_lgbm)
    model_lgbm = LGBM.Model(config_lgbm)

    # Create XGB Model
    config_xgb = XGB.Config(**best_params_xgb)
    model_xgb = XGB.Model(config_xgb)

    # Train LGBM
    start_lgbm = time.time()
    model_lgbm.fit(X, y)
    end_lgbm = time.time()
    print(f'LGBM Training time: {end_lgbm - start_lgbm} seconds')

    # Train XGB
    start_xgb = time.time()
    model_xgb.fit(X, y)
    end_xgb = time.time()
    print(f'XGB Training time: {end_xgb - start_xgb} seconds')

    print('Best hyperparameters for LGBM:', best_params_lgbm)
    print('Best hyperparameters for XGB:', best_params_xgb)

    # Predict with LGBM
    y_pred_lgbm = model_lgbm.predict(test)

    # Predict with XGB
    y_pred_xgb = model_xgb.predict(test)

    # Inverse map labels for LGBM
    y_pred_lgbm = inverse_map_labels(pd.DataFrame({const.TARGET_COL: y_pred_lgbm}))[const.TARGET_COL]

    # Inverse map labels for XGB
    y_pred_xgb = inverse_map_labels(pd.DataFrame({const.TARGET_COL: y_pred_xgb}))[const.TARGET_COL]

    # Ensemble predictions
    ensemble_predictions = [y_pred_lgbm, y_pred_xgb]
    final_predictions = []
    for i in range(len(y_pred_lgbm)):
        row = [ensemble_predictions[j][i] for j in range(len(ensemble_predictions))]
        counts = {}
        for label in row:
            counts[label] = counts.get(label, 0) + 1
        if len(counts) > 1:
            print(i, row, counts)
        final_predictions.append(max(counts, key=counts.get))

    # Save submission
    sample_submission_df[const.TARGET_COL] = final_predictions
    sample_submission_df.to_csv(f'submission_{int(time.time())}.csv', index=False)

if __name__ == '__main__':
    main()

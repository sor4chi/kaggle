import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import constant as const
# from models.lightgbm_classifier import model as XGB
from models.xgboost_classifier import model as XGB
from processor import preprocess, inverse_map_labels

def main():
    train_df = pd.read_csv(const.TRAIN_CSV_PATH, index_col='id')
    test_df = pd.read_csv(const.TEST_CSV_PATH, index_col='id')
    sample_submission_df = pd.read_csv(const.SAMPLE_SUBMISSION_CSV_PATH)

    # Preprocessing
    X, y, test = preprocess(train_df, test_df)

    # Hiperparameter optimization
    best_params = XGB.optimize(X, y)

    print('Best hyperparameters:', best_params)

    # Create Model
    config = XGB.Config(**best_params)
    model = XGB.Model(config)

    # Train
    start = time.time()
    model.fit(X, y)
    end = time.time()
    print(f'Training time: {end - start} seconds')


    # Predict
    y_pred = model.predict(test)

    # Inverse map labels
    y_pred = inverse_map_labels(pd.DataFrame({const.TARGET_COL: y_pred}))[const.TARGET_COL]

    # Save submission
    sample_submission_df[const.TARGET_COL] = y_pred
    sample_submission_df.to_csv(f'submission_{int(time.time())}.csv', index=False)

if __name__ == '__main__':
    main()

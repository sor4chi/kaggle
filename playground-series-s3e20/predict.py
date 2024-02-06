from resources import TARGET_COLUMN, PATHS, SELECTED_FEATURES, predictionData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

pkl_list = os.listdir(PATHS['model_dir'])
selected_model = pkl_list[0]
for i, pkl in enumerate(pkl_list):
    print(i, pkl)
selected_model = pkl_list[int(input('Select model : '))]
path = os.path.join(PATHS['model_dir'], selected_model)

# 予測モデルの読み込み
with open(path, mode='rb') as f:
    reg_lgb = pickle.load(f)

# predictionDataの特徴量選択
predictionData = predictionData[SELECTED_FEATURES]

# predictionDataの欠損値補完
predictionData = predictionData.fillna(predictionData.median())

# 予測
y_pred = reg_lgb.predict(predictionData)

# 予測結果が負になる場合は0に置換
y_pred = np.where(y_pred < 0, 0, y_pred)

# 提出用ファイルの作成
submission = pd.DataFrame({'ID_LAT_LON_YEAR_WEEK': predictionData.index, TARGET_COLUMN: y_pred})
submission.to_csv(PATHS['prediction']('lgb'), index=False)

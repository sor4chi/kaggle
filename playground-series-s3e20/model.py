from resources import TARGET_COLUMN, PATHS, SELECTED_FEATURES, trainData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

pd.set_option('display.max_rows', 1000)

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# 欠損値の補完
trainData = trainData.fillna(trainData.median())

# Test/Trainの分割
X_train, X_test, y_train, y_test = train_test_split(trainData[SELECTED_FEATURES], trainData[TARGET_COLUMN], test_size=0.2, random_state=42)

# LightGBMのモデル作成
lgb_model = lgb.LGBMRegressor()
reg_lgb = GridSearchCV(lgb_model,
                        {'max_depth': [2, 3, 4],
                        'n_estimators': [50, 100, 200]},
                        verbose=1)
reg_lgb.fit(X_train, y_train)

# テストデータでの予測
y_pred = reg_lgb.predict(X_test)

# RMSEの計算
rmse = np.sqrt(MSE(y_test, y_pred))
print('RMSE :', rmse)

# 予測モデルの保存
with open(PATHS['model']('lgb'), mode='wb') as f:
    pickle.dump(reg_lgb, f)

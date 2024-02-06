from resources import TARGET_COLUMN, PATHS, SELECTED_FEATURES, trainData
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pickle
import os

# latitudeとlongitudeから外接する矩形を計算
def get_boundary(lat, lon):
    lat_max = lat.max()
    lat_min = lat.min()
    lon_max = lon.max()
    lon_min = lon.min()
    return lat_max, lat_min, lon_max, lon_min

X_min, X_max, Z_min, Z_max = get_boundary(trainData['latitude'], trainData['longitude'])

# X_min, X_max, Z_min, Z_max を使って、3次元のグラフを描画
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(trainData['latitude'], trainData['longitude'], trainData[TARGET_COLUMN], s=0.1)
ax.set_xlabel('latitude')
ax.set_ylabel('longitude')
ax.set_zlabel(TARGET_COLUMN)
ax.set_xlim(X_min, X_max)
ax.set_ylim(Z_min, Z_max)
plt.show()

IS_lOG_SCALE = True

# 毎週ごとにグラフを作成し、matplotlib.animationでアニメーションを作成
plots = []
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for year in range(2019, 2022):
    for week in range(1, 53):
        trainData_week = trainData[(trainData['week_no'] == week) & (trainData['year'] == year)]
        target = (trainData_week[TARGET_COLUMN] + 1).apply(np.log) if IS_lOG_SCALE else trainData_week[TARGET_COLUMN]
        plot = ax.plot_trisurf(trainData_week['latitude'], trainData_week['longitude'], target, cmap='viridis', edgecolor='none')
        ax.set_xlabel('latitude')
        ax.set_ylabel('longitude')
        ax.set_zlabel(TARGET_COLUMN)
        label = ax.text2D(0.05, 0.95, f'year : {year}, week : {week}', transform=ax.transAxes)
        plots.append([plot, label])

ani = animation.ArtistAnimation(fig, plots, interval=100)
ani.save(PATHS['plot']('log' if IS_lOG_SCALE else 'linear'), writer='pillow')
plt.show()


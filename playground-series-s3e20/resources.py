import time
import pandas as pd

PATHS = {
    'train': './data/train.csv',
    'test': './data/test.csv',
    'sample_submission': './data/sample_submission.csv',
    'prediction': lambda model_name: f'./prediction/{model_name}_{time.strftime("%Y%m%d%H%M%S")}.csv',
    'model_dir': './model',
    'model': lambda model_name: f'{PATHS["model_dir"]}/{model_name}_{time.strftime("%Y%m%d%H%M%S")}.pkl',
    'plot': lambda model_name: f'./plot/{model_name}_{time.strftime("%Y%m%d%H%M%S")}.gif',
}

TARGET_COLUMN = 'emission'

trainData = pd.read_csv(PATHS['train'], index_col='ID_LAT_LON_YEAR_WEEK')
predictionData = pd.read_csv(PATHS['test'], index_col='ID_LAT_LON_YEAR_WEEK')

# とりあえず、全ての特徴量を使って学習
# SELECTED_FEATURES = [col for col in trainData.columns if col != TARGET_COLUMN]

# 以下の特徴量を使って学習
SELECTED_FEATURES = [
    'latitude',
    'longitude',
    'year',
    'week_no',
    'emission',
]

# Column names: 日本語の説明
# latitude: 緯度
# longitude: 経度
# year: 観測年
# week_no: 観測週

# SulphurDioxide_SO2_column_number_density: 二酸化硫黄の濃度
# SulphurDioxide_SO2_column_number_density_amf: 二酸化硫黄の濃度のアンサンブル平均
# SulphurDioxide_SO2_slant_column_number_density: 二酸化硫黄の濃度のスラント列密度
# SulphurDioxide_cloud_fraction: 雲の割合
# SulphurDioxide_sensor_azimuth_angle: センサーの方位角
# SulphurDioxide_sensor_zenith_angle: センサーの天頂角
# SulphurDioxide_solar_azimuth_angle: 太陽の方位角
# SulphurDioxide_solar_zenith_angle: 太陽の天頂角
# SulphurDioxide_SO2_column_number_density_15km: 二酸化硫黄の濃度の15kmの高さでの密度

# CarbonMonoxide_CO_column_number_density： 一酸化炭素の濃度
# CarbonMonoxide_H2O_column_number_density: 水蒸気の濃度
# CarbonMonoxide_cloud_height: 雲の高さ
# CarbonMonoxide_sensor_altitude: センサーの高さ
# CarbonMonoxide_sensor_azimuth_angle: センサーの方位角
# CarbonMonoxide_sensor_zenith_angle: センサーの天頂角
# CarbonMonoxide_solar_azimuth_angle: 太陽の方位角
# CarbonMonoxide_solar_zenith_angle: 太陽の天頂角

# NitrogenDioxide_NO2_column_number_density: 二酸化窒素の濃度
# NitrogenDioxide_tropospheric_NO2_column_number_density: 二酸化窒素の濃度の対流圏の密度
# NitrogenDioxide_stratospheric_NO2_column_number_density: 二酸化窒素の濃度の成層圏の密度
# NitrogenDioxide_NO2_slant_column_number_density: 二酸化窒素の濃度のスラント列密度
# NitrogenDioxide_tropopause_pressure: 対流圏の圧力
# NitrogenDioxide_absorbing_aerosol_index: 吸収性エアロゾル指数
# NitrogenDioxide_cloud_fraction: 雲の割合
# NitrogenDioxide_sensor_altitude: センサーの高さ
# NitrogenDioxide_sensor_azimuth_angle: センサーの方位角
# NitrogenDioxide_sensor_zenith_angle: センサーの天頂角
# NitrogenDioxide_solar_azimuth_angle: 太陽の方位角
# NitrogenDioxide_solar_zenith_angle: 太陽の天頂角

# Formaldehyde_tropospheric_HCHO_column_number_density: ホルムアルデヒドの濃度の対流圏の密度
# Formaldehyde_tropospheric_HCHO_column_number_density_amf: ホルムアルデヒドの濃度の対流圏の密度のアンサンブル平均
# Formaldehyde_HCHO_slant_column_number_density: ホルムアルデヒドの濃度のスラント列密度
# Formaldehyde_cloud_fraction: 雲の割合
# Formaldehyde_solar_zenith_angle: 太陽の天頂角
# Formaldehyde_solar_azimuth_angle: 太陽の方位角
# Formaldehyde_sensor_zenith_angle: センサーの天頂角
# Formaldehyde_sensor_azimuth_angle: センサーの方位角

# UvAerosolIndex_absorbing_aerosol_index: 吸収性エアロゾル指数
# UvAerosolIndex_sensor_altitude: センサーの高さ
# UvAerosolIndex_sensor_azimuth_angle: センサーの方位角
# UvAerosolIndex_sensor_zenith_angle: センサーの天頂角
# UvAerosolIndex_solar_azimuth_angle: 太陽の方位角
# UvAerosolIndex_solar_zenith_angle: 太陽の天頂角

# Ozone_O3_column_number_density: オゾンの濃度
# Ozone_O3_column_number_density_amf: オゾンの濃度のアンサンブル平均
# Ozone_O3_slant_column_number_density: オゾンの濃度のスラント列密度
# Ozone_O3_effective_temperature: オゾンの効果温度
# Ozone_cloud_fraction: 雲の割合
# Ozone_sensor_azimuth_angle: センサーの方位角
# Ozone_sensor_zenith_angle: センサーの天頂角
# Ozone_solar_azimuth_angle: 太陽の方位角
# Ozone_solar_zenith_angle: 太陽の天頂角

# UvAerosolLayerHeight_aerosol_height: エアロゾルの高さ
# UvAerosolLayerHeight_aerosol_pressure: エアロゾルの圧力
# UvAerosolLayerHeight_aerosol_optical_depth: エアロゾルの光学的厚さ
# UvAerosolLayerHeight_sensor_zenith_angle: センサーの天頂角
# UvAerosolLayerHeight_sensor_azimuth_angle: センサーの方位角
# UvAerosolLayerHeight_solar_azimuth_angle: 太陽の方位角
# UvAerosolLayerHeight_solar_zenith_angle: 太陽の天頂角

# Cloud_cloud_fraction: 雲の割合
# Cloud_cloud_top_pressure: 雲の上の圧力
# Cloud_cloud_top_height: 雲の上の高さ
# Cloud_cloud_base_pressure: 雲の下の圧力
# Cloud_cloud_base_height: 雲の下の高さ
# Cloud_cloud_optical_depth: 雲の光学的厚さ
# Cloud_surface_albedo: 地表面のアルベド
# Cloud_sensor_azimuth_angle: センサーの方位角
# Cloud_sensor_zenith_angle: センサーの天頂角
# Cloud_solar_azimuth_angle: 太陽の方位角
# Cloud_solar_zenith_angle: 太陽の天頂角

# emission: 二酸化炭素の排出量


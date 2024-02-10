import pandas as pd
import pyjson5
import constant as const
from utils import get_mapping
from sklearn.preprocessing import OneHotEncoder


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    MAPPING = get_mapping()
    mapping = {}
    for column in df.columns:
        if column in MAPPING:
            mapping[column] = MAPPING[column]
            df[column] = df[column].map(MAPPING[column])
    return df

def inverse_map_labels(df: pd.DataFrame) -> pd.DataFrame:
    MAPPING = get_mapping()
    inverse_mapping = {}
    for column in df.columns:
        if column in MAPPING:
            inverse_mapping[column] = {v: k for k, v in MAPPING[column].items()}
            df[column] = df[column].map(inverse_mapping[column])
    return df

def preprocess_mapping(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Replace string labels with numeric values
    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    X = train_df.drop(columns=[const.TARGET_COL])
    y = train_df[const.TARGET_COL]
    return X, y, test_df

def preprocess_one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # const.CATEGORICAL_COLUMNS
    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X = train_df.drop(columns=[const.TARGET_COL])
    y = train_df[const.TARGET_COL]
    test = test_df
    for column in const.CATEGORICAL_COLUMNS:
        encoder.fit(train_df[[column]])
        X = pd.concat([X, pd.DataFrame(encoder.transform(train_df[[column]]), columns=encoder.get_feature_names_out([column]))], axis=1)
        test = pd.concat([test, pd.DataFrame(encoder.transform(test_df[[column]]), columns=encoder.get_feature_names_out([column]))], axis=1)
    X = X.drop(columns=const.CATEGORICAL_COLUMNS)
    test = test.drop(columns=const.CATEGORICAL_COLUMNS)
    return X, y, test

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X, y, test = preprocess_mapping(train_df, test_df)
    return X, y, test

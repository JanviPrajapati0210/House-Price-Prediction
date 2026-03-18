import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def drop_useless_columns(df):
    # these have 90%+ missing, not worth keeping
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df

def fill_missing_values(df):
    # missing here means the feature doesnt exist, not actually unknown
    none_cols = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinType2', 'GarageType',
                 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu']
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # numerical - fill with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # categorical - fill with most common value
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def feature_engineering(df):
    # age of house when sold
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # years since last remodel
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

    # total sqft - combining all floors and basement
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # total bathrooms across whole house
    df['TotalBaths'] = (df['FullBath'] + df['HalfBath'] * 0.5 +
                        df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5)

    # all porch types combined
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                          df['3SsnPorch'] + df['ScreenPorch'])

    # simple binary flags
    df['HasGarage']   = (df['GarageArea'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasPool']     = (df['PoolArea'] > 0).astype(int)

    # quality x area interaction - usually very predictive
    df['QualityXArea'] = df['OverallQual'] * df['GrLivArea']

    return df

def encode_categorical(df):
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def preprocess(train_path, test_path):
    train, test = load_data(train_path, test_path)

    # separate target before anything
    y = train['SalePrice']
    train = train.drop(columns=['SalePrice'])

    # combine train and test so encoding is consistent
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)
    combined = drop_useless_columns(combined)
    combined = fill_missing_values(combined)
    combined = feature_engineering(combined)  # 👈 now actually called
    combined = encode_categorical(combined)

    # split back
    X_train = combined.iloc[:len(train), :]
    X_test  = combined.iloc[len(train):, :]

    return X_train, X_test, y

if __name__ == "__main__":
    X_train, X_test, y = preprocess('../data/train.csv', '../data/test.csv')
    print("preprocessing done!")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y       : {y.shape}")
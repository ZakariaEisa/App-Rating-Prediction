import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats
import Preprocessing
Seed=17


def main():
    train = pd.read_csv('train.csv').drop_duplicates().reset_index(drop=True)
    test = pd.read_csv('test.csv')
    submission = pd.read_csv('SampleSubmission.csv')
    ref_date = pd.Timestamp('2025-04-24')

    X = Preprocessing.preprocess(train, ref_date)
    y = train['Y'].fillna(train['Y'].median())
    X_test = Preprocessing.preprocess(test, ref_date)

    X, y = Preprocessing.remove_outliers(X, y)

    te_cols = [c for c in X.columns if c.endswith('_Enc')]
    X, X_test = Preprocessing.smooth_target_encode(X, X_test, y, te_cols)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
    X_val[num_feats] = scaler.transform(X_val[num_feats])
    X_test[num_feats] = scaler.transform(X_test[num_feats])

    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=16,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=Seed
    )

    lgb.fit(X_train, y_train)

    def blend(preds, tgt=4.25, alpha=0.9):
        b = (1 - alpha) * preds + alpha * tgt
        return np.clip(b, 4.2, 4.35)
    y_pred_train=lgb.predict(X_train)
    y_pred_val = lgb.predict(X_val)
    print("Train MAE (lgb model):", mean_absolute_error(y_train, y_pred_train))
    print("Validation MAE (lgb model):", mean_absolute_error(y_val, y_pred_val))

    y_pred_test = blend(lgb.predict(X_test))
    submission['Y'] = y_pred_test
    submission.to_csv('0FinalP(lgb model).csv', index=False)
    print("✅ Saved 0FinalP (lgb model).csv")

if __name__ == "__main__":
    main()

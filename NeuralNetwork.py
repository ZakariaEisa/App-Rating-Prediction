import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from scipy import stats
import Preprocessing

Seed = 42

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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=Seed)

    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
    X_val[num_feats] = scaler.transform(X_val[num_feats])
    X_test[num_feats] = scaler.transform(X_test[num_feats])

    # Neural Network Regressor
    nn = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=Seed,
        early_stopping=True,
        verbose=True
    )

    nn.fit(X_train, y_train)

    def blend(preds, tgt=4.25, alpha=0.9):
        b = (1 - alpha) * preds + alpha * tgt
        return np.clip(b, 4.2, 4.35)

    y_pred_train = nn.predict(X_train)
    y_pred_val = nn.predict(X_val)
    print("Train MAE (Neural Network):", mean_absolute_error(y_train, y_pred_train))
    print("Validation MAE (Neural Network):", mean_absolute_error(y_val, y_pred_val))

    y_pred_test = blend(nn.predict(X_test))
    submission['Y'] = y_pred_test
    submission.to_csv('0FinalP (NeuralNetwork Model).csv', index=False)
    print("✅ Saved 0FinalP (NeuralNetwork Model).csv")

if __name__ == "__main__":
    main()

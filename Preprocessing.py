import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats

def preprocess(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy().drop(columns=['Y'], errors='ignore')

    df = df.rename(columns={
        'X0': 'App_Name', 'X1': 'Category', 'X2': 'Reviews', 'X3': 'Size',
        'X4': 'Installs', 'X5': 'Free_Paid', 'X6': 'Price', 'X7': 'Content_Rating',
        'X8': 'Sub_Category', 'X9': 'Last_Updated', 'X10': 'Current_Version', 'X11': 'Android_Version'
    })

    df['Free_Paid_Num'] = df['Free_Paid'].map({'Free': 0, 'Paid': 1}).fillna(0).astype(int)
    df['Title_Length'] = df['App_Name'].astype(str).str.len()
    df['Title_Avg_Word_Length'] = df['App_Name'].str.split().apply(lambda ws: np.mean([len(w) for w in ws]) if ws else 0)

    def parse_size(sz):
        if isinstance(sz, str):
            s = sz.strip().upper()
            if s.endswith('M'): return float(s[:-1]) * 1e6
            if s.endswith('K'): return float(s[:-1]) * 1e3
            if s == 'VARY WITH DEVICE': return np.nan
        return np.nan
    df['Size_Num'] = np.log1p(df['Size'].apply(parse_size).fillna(df['Size'].apply(parse_size).median()))

    df['Reviews_Num'] = df['Reviews'].astype(str).str.replace(r'\D', '', regex=True).replace('', '0').astype(int)
    df['Installs_Num'] = df['Installs'].astype(str).str.replace(r'\D', '', regex=True).replace('', '0').astype(int)
    df['Log_Installs'] = np.log1p(df['Installs_Num'])

    df['Price_Num'] = (
        df['Price'].astype(str)
                 .str.replace(r'[\$,]', '', regex=True)
                 .apply(lambda x: float(x) if re.match(r'^\d+(\.\d+)?$', x) else 0.0)
    )
    df['Log_Price'] = np.log1p(df['Price_Num'])
    df['Is_Free'] = (df['Price_Num'] == 0).astype(int)

    df['Review_Install_Ratio'] = df['Reviews_Num'] / (df['Installs_Num'] + 1)
    df['Install_Size_Ratio'] = df['Installs_Num'] / (df['Size_Num'] + 1)
    df['Price_Size_Interaction'] = df['Price_Num'] * df['Size_Num']

    df['Last_Updated_DT'] = pd.to_datetime(df['Last_Updated'], errors='coerce')
    df['Days_Since_Update'] = (reference_date - df['Last_Updated_DT']).dt.days.fillna(0)
    df['Updated_Recently_90_Days'] = (df['Days_Since_Update'] <= 90).astype(int)
    df['LogInstalls_Per_Day'] = df['Log_Installs'] / (df['Days_Since_Update'] + 1)



    for col in ['Category', 'Content_Rating']:
        df[col] = df[col].fillna('MISSING').astype(str)
        df[col + '_Enc'] = LabelEncoder().fit_transform(df[col])

    drop_cols = [
        'App_Name', 'Size', 'Reviews', 'Installs', 'Price', 'Free_Paid',
        'Last_Updated', 'Last_Updated_DT', 'Current_Version', 'Android_Version',
        'Category', 'Content_Rating', 'Sub_Category',"Installs_Num"
    ]
    return df.drop(columns=drop_cols)

def remove_outliers(X: pd.DataFrame, y: pd.Series, z_threshold=3):
    mask = np.abs(stats.zscore(y)) < z_threshold
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

def smooth_target_encode(train_df, test_df, y, cols, alpha=5):
    global_mean = y.mean()
    for col in cols:
        tmp = pd.DataFrame({col: train_df[col], 'target': y})
        agg = tmp.groupby(col)['target'].agg(['mean', 'count'])
        smooth = (agg['mean'] * agg['count'] + global_mean * alpha) / (agg['count'] + alpha)
        train_df[col + '_TE'] = train_df[col].map(smooth)
        test_df[col + '_TE'] = test_df[col].map(smooth).fillna(global_mean)
        train_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)
    return train_df, test_df

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
import utils
import wandb
import sys

# 데이터프레임 불러오기
def fit(data, model, ignore_wandb, mask_ratio):
    df = data
    df = df[['cluster', 'age', 'CT_R', 'CT_E',
        'dis', 'danger','gender', 'is_korean', 'primary case', 'job_idx', 'place_idx', 'add_idx',
        'rep_idx',  'diff_days', 'y']]
    df['cluster'] = df['cluster'].map({value: idx for idx, value in enumerate(sorted(df['cluster'].unique()))})
    grouped = df.groupby('cluster')['diff_days'].agg(['max', 'min'])
    df['d'] = df['cluster'].map(lambda x: (grouped.loc[x, 'max'] - grouped.loc[x, 'min']) + 1)

    continuous_cols = df.columns[1:6]
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    categorical_cols = df.columns[6:12]
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(df[categorical_cols]).toarray()
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(categorical_cols))], axis=1)

    y_col = df.pop('y')
    d_col = df.pop('d')
    df['y'] = y_col
    df['d'] = d_col
    
    df = utils.delete_df_rows_by_ratio(df, mask_ratio) 
    grouped = df.groupby('cluster').mean().reset_index()

    # grouped 데이터프레임에서 X와 y를 추출
    X = grouped.drop(['y', 'd','cluster', 'diff_days'], axis=1)  # 입력 특성 (y, d 제외)

    y = grouped[['y','d']]

    # 학습 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model == "svr":
        base = SVR()
    elif model == "rfr":
        base = RandomForestRegressor()
    momodel = MultiOutputRegressor(base)
    momodel.fit(X_train, y_train)

    # 예측 수행
    y_pred_tr = momodel.predict(X_train)
    y_pred = momodel.predict(X_test)
    # MAE, RMSE 평가 및 출력
    tr_mse = mean_squared_error(y_train, y_pred_tr)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("MAE: ", mae)
    print("RMSE: ", rmse)
    # 전체 데이터에 대한 예측 결과 출력
    # result = pd.DataFrame({'cluster': X.index, 'y_true': y['y'], 'y_pred': y_pred[:,0], 'd_true': y['d'], 'd_pred': y_pred[:,1]})
    if not ignore_wandb:
        wandb.log({"tr_loss" : tr_mse,
                "te_mae_loss" : mae,
                "te_rmse_loss" : rmse,
                })
    # print(result)
    sys.exit()

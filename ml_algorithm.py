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
def fit(data, model, ignore_wandb, date_cutoff):
    data = data[['cluster', 'age', 'CT_R', 'CT_E',
        'dis', 'danger','gender', 'is_korean', 'primary case', 'job_idx', 'place_idx', 'add_idx',
        'rep_idx',  'diff_days', 'y']]
    data['cluster'] = data['cluster'].map({value: idx for idx, value in enumerate(sorted(data['cluster'].unique()))})

    min_diff_days = data.groupby('cluster')['diff_days'].min()
    data['diff_days'] = data.apply(lambda row: row['diff_days'] - min_diff_days[row['cluster']], axis=1)
    diff_max_group = data.groupby('cluster')['diff_days'].max()
    data['d'] = data['cluster'].map(lambda x: diff_max_group[x] + 1)
    data = data.sort_values(by=['cluster', 'diff_days'], ascending=[True, True])
    data.drop(data[(data['d'] <= date_cutoff) | (data['diff_days'] > date_cutoff)].index, inplace=True)
    data['cluster'] = data['cluster'].map({value: idx for idx, value in enumerate(sorted(data['cluster'].unique()))})
    continuous_cols = data.columns[1:6]
    scaler_x = StandardScaler()
    data[continuous_cols] = scaler_x.fit_transform(data[continuous_cols])
    scaler_y = StandardScaler()
    data[['d','y']] = scaler_y.fit_transform(data[['d', 'y']])
    categorical_cols = data.columns[6:12]
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(data[categorical_cols]).toarray()
    data = data.drop(categorical_cols, axis=1)
    x=pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)
    data = pd.concat([data, pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)], axis=1)
    # data.to_csv('temp.csv')
    y_col = data.pop('y')
    d_col = data.pop('d')
    data['y'] = y_col
    data['d'] = d_col
    
    # data = utils.delete_data_rows_by_ratio(data, mask_ratio) 
    
    grouped = data.groupby('cluster').mean().reset_index()

    # grouped 데이터프레임에서 X와 y를 추출
    X = grouped.drop(['y', 'd','cluster', 'diff_days'], axis=1)  # 입력 특성 (y, d 제외)

    y = grouped[['y','d']]
    # 학습 데이터와 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if model == "svr":
        base = SVR()
    elif model == "rfr":
        base = RandomForestRegressor()
    mo_model = MultiOutputRegressor(base)
    mo_model.fit(X_train, y_train)

    # 예측 수행
    y_pred_tr = mo_model.predict(X_train)
    y_pred = mo_model.predict(X_test)
    y_pred_restored = scaler_y.inverse_transform(y_pred.reshape(-1, 2))
    y_true_restored = scaler_y.inverse_transform(y_test.values.reshape(-1, 2))
    # MAE, RMSE 평가 및 출력
    tr_mse = mean_squared_error(y_train, y_pred_tr)
    mae_y = mean_absolute_error(y_true_restored[:,0], y_pred_restored[:,0])
    rmse_y = np.sqrt(mean_squared_error(y_true_restored[:,0], y_pred_restored[:,0]))
    
    mae_d = mean_absolute_error(y_true_restored[:,1], y_pred_restored[:,1])
    rmse_d = np.sqrt(mean_squared_error(y_true_restored[:,1], y_pred_restored[:,1]))
    print("MAE (y): ", mae_y)
    print("RMSE (y): ", rmse_y)
    print("MAE (d): ", mae_d)
    print("RMSE (d): ", rmse_d)    
    # 전체 데이터에 대한 예측 결과 출력
    # result = pd.DataFrame({'cluster': X.index, 'y_true': y['y'], 'y_pred': y_pred[:,0], 'd_true': y['d'], 'd_pred': y_pred[:,1]})
    if not ignore_wandb:
        wandb.log({"tr_loss" : tr_mse,
                "te_mae_loss (y)" : mae_y,
                "te_rmse_loss (y)" : rmse_y,
                "te_mae_loss (d)" : mae_d,
                "te_rmse_loss (d)" : rmse_d,
                })
    # print(result)
    sys.exit()

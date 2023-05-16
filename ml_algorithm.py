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
def fit(data, model, ignore_wandb, print_idx):
    data = data[['cluster', 'age', 'CT_R', 'CT_E',
        'dis', 'danger','gender', 'is_korean', 'primary case', 'job_idx', 'place_idx', 'add_idx',
        'rep_idx',  'diff_days', 'y']]
    data['cluster'] = data['cluster'].map({value: idx for idx, value in enumerate(sorted(data['cluster'].unique()))})

    data = data.sort_values(by=['cluster', 'diff_days'], ascending=[True, True])
    min_diff_days = data.groupby('cluster')['diff_days'].min()
    data['diff_days'] = data.apply(lambda row: row['diff_days'] - min_diff_days[row['cluster']], axis=1)
    grouped = data.groupby('cluster')['diff_days'].max()
    data['d'] = data.apply(lambda row: grouped[row['cluster']] - row['diff_days'], axis=1)
    for _, group_data in data.groupby('cluster'):
        for index, row in group_data.iterrows():
            group_data.loc[index, 'y'] = group_data[group_data['diff_days'] > row['diff_days']].shape[0]
        data.update(group_data)
    data['cut_date'] = data['cluster'].map(lambda x: grouped[x] + 1)
    datasets=[]; sclaers = {}
    for i in range(1,6):
        cutted_data=data.drop(data[(data['cut_date'] < i) | (data['diff_days'] >= i)].index)
        cutted_data['cluster'] = cutted_data['cluster'].map({value: idx for idx, value in enumerate(sorted(cutted_data['cluster'].unique()))})
        continuous_cols = cutted_data.columns[1:6]
        sclaers[f'x_{i}'] = StandardScaler()
        cutted_data[continuous_cols] = sclaers[f'x_{i}'].fit_transform(cutted_data[continuous_cols])
        sclaers[f'd_{i}'] = StandardScaler()
        sclaers[f'y_{i}'] = StandardScaler()
        cutted_data[['d']] = sclaers[f'd_{i}'].fit_transform(cutted_data[['d']])
        cutted_data[['y']] = sclaers[f'y_{i}'].fit_transform(cutted_data[['y']])
        categorical_cols = cutted_data.columns[6:12]
        encoder = OneHotEncoder()
        one_hot = encoder.fit_transform(cutted_data[categorical_cols]).toarray()
        cutted_data = cutted_data.drop(categorical_cols, axis=1)
        cutted_data = pd.concat([cutted_data, pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(categorical_cols), index=cutted_data.index)], axis=1)
        y_col = cutted_data.pop('y')
        d_col = cutted_data.pop('d')
        cutted_data['y'] = y_col
        cutted_data['d'] = d_col
        meancut_data = cutted_data.groupby('cluster').mean().reset_index()

        datasets.append(meancut_data)
    # import pdb;pdb.set_trace()
    X_trains = []; X_tests = []; Y_trains =[]; Y_tests=[]
    for dataset in datasets:
        X = dataset.drop(['y', 'd','cluster', 'diff_days'], axis=1) 
        y = dataset[['y','d']]
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        X_trains.append(X_train); X_tests.append(X_test); Y_trains.append(Y_train); Y_tests.append(Y_test)

    X_trainset = pd.concat(X_trains); Y_trainset = pd.concat(Y_trains)
    X_testset = pd.concat(X_tests); Y_testset = pd.concat(Y_tests)
    
    if model == "svr":
        base = SVR()
    elif model == "rfr":
        base = RandomForestRegressor()
    if print_idx == 0:
        X_test = X_testset
        y_test = Y_testset
    else:
        X_test = X_tests[print_idx]
        y_test = Y_tests[print_idx]

    mo_model = MultiOutputRegressor(base)
    mo_model.fit(X_trainset, Y_trainset)

    # 예측 수행
    y_pred_tr = mo_model.predict(X_trainset)
    y_pred = mo_model.predict(X_test)
    y_pred_restored = scaler_y.inverse_transform(y_pred)[:,0]
    y_true_restored = scaler_y.inverse_transform(y_test.values)[:,0]
    d_pred_restored = scaler_d.inverse_transform(y_pred)[:,1]
    d_true_restored = scaler_d.inverse_transform(y_test.values)[:,1]
    
    # MAE, RMSE 평가 및 출력
    tr_mse = mean_squared_error(Y_trainset, y_pred_tr)
    mae_y = mean_absolute_error(y_true_restored, y_pred_restored)
    rmse_y = np.sqrt(mean_squared_error(y_true_restored, y_pred_restored))
    
    mae_d = mean_absolute_error(d_true_restored, d_pred_restored)
    rmse_d = np.sqrt(mean_squared_error(d_true_restored, d_pred_restored))
    print("MAE (d): ", mae_d)
    print("MAE (y): ", mae_y)
    print("RMSE (d): ", rmse_d)    
    print("RMSE (y): ", rmse_y)
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

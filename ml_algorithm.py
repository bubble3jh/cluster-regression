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
    X_trainset = pd.concat(X_trains, ignore_index=True); Y_trainset = pd.concat(Y_trains, ignore_index=True); X_trainset=X_trainset.fillna(0)
    X_testset = pd.concat(X_tests, ignore_index=True); Y_testset = pd.concat(Y_tests, ignore_index=True); X_testset=X_testset.fillna(0)
    X_testset['add_idx_8']=0
    # import pdb;pdb.set_trace()
    if model == "svr":
        base = SVR()
    elif model == "rfr":
        base = RandomForestRegressor()
    if print_idx == 0:
        X_test = X_testset
        y_test = Y_testset
    else:
        X_test = X_tests[print_idx-1]
        y_test = Y_tests[print_idx-1]
        # X_test['add_idx_8']=0
        # X_test['add_idx_25']=0
        # X_test['add_idx_29']=0
        X_test=X_test[['age', 'CT_R', 'CT_E', 'dis', 'danger', 'rep_idx', 'cut_date',
       'gender_0', 'gender_1', 'is_korean_0', 'is_korean_1', 'primary case_0',
       'primary case_1', 'job_idx_0', 'job_idx_1', 'job_idx_2', 'job_idx_3',
       'job_idx_4', 'job_idx_5', 'job_idx_6', 'job_idx_7', 'job_idx_8',
       'job_idx_9', 'job_idx_99', 'place_idx_1', 'place_idx_2', 'place_idx_3',
       'place_idx_4', 'place_idx_5', 'place_idx_6', 'place_idx_7',
       'place_idx_8', 'place_idx_9', 'place_idx_10', 'place_idx_11',
       'place_idx_12', 'place_idx_13', 'place_idx_14', 'place_idx_15',
       'place_idx_16', 'place_idx_17', 'place_idx_18', 'place_idx_19',
       'add_idx_0', 'add_idx_1', 'add_idx_2', 'add_idx_3', 'add_idx_4',
       'add_idx_5', 'add_idx_6', 'add_idx_7', 'add_idx_8', 'add_idx_9',
       'add_idx_10', 'add_idx_11', 'add_idx_12', 'add_idx_13', 'add_idx_14',
       'add_idx_15', 'add_idx_16', 'add_idx_17', 'add_idx_18', 'add_idx_19',
       'add_idx_20', 'add_idx_21', 'add_idx_22', 'add_idx_23', 'add_idx_24',
       'add_idx_25', 'add_idx_26', 'add_idx_27', 'add_idx_28', 'add_idx_29',
       'add_idx_30']]
        # import pdb;pdb.set_trace()
    mo_model = MultiOutputRegressor(base)
    mo_model.fit(X_trainset, Y_trainset)

    # 예측 수행
    y_pred_tr = mo_model.predict(X_trainset)
    y_pred = mo_model.predict(X_test)
    y_pred_restored = sclaers[f'y_{print_idx}'].inverse_transform(y_pred)[:,0]
    y_true_restored = sclaers[f'y_{print_idx}'].inverse_transform(y_test.values)[:,0]
    d_pred_restored = sclaers[f'd_{print_idx}'].inverse_transform(y_pred)[:,1]
    d_true_restored = sclaers[f'd_{print_idx}'].inverse_transform(y_test.values)[:,1]
    # print(d_pred_restored, d_true_restored)
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

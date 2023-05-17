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
def fit(data_path, model, ignore_wandb, cutdates_num, print_idx):
    datasets=[]; sclaers = {}
    for i in range(cutdates_num+1):
        data = pd.read_csv(data_path+f"data_cut_{i}.csv")
        continuous_cols = data.columns[1:6]
        sclaers[f'x_{i}'] = StandardScaler()
        data[continuous_cols] = sclaers[f'x_{i}'].fit_transform(data[continuous_cols])
        sclaers[f'd_{i}'] = StandardScaler()
        sclaers[f'y_{i}'] = StandardScaler()
        data[['d']] = sclaers[f'd_{i}'].fit_transform(data[['d']])
        data[['y']] = sclaers[f'y_{i}'].fit_transform(data[['y']])
        categorical_cols = data.columns[6:12]
        encoder = OneHotEncoder()
        one_hot = encoder.fit_transform(data[categorical_cols]).toarray()
        data = data.drop(categorical_cols, axis=1)
        data = pd.concat([data, pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)], axis=1)
        
        y_col = data.pop('y')
        d_col = data.pop('d')
        data['y'] = y_col
        data['d'] = d_col
        meancut_data = data.groupby('cluster').mean().reset_index()

        datasets.append(meancut_data)
    # import pdb;pdb.set_trace()
    X_trains = []; X_tests = []; Y_trains =[]; Y_tests=[]
    for dataset in datasets:
        X = dataset.drop(['y', 'd','cluster', 'diff_days'], axis=1) 
        y = dataset[['y','d']]
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        X_trains.append(X_train); X_tests.append(X_test); Y_trains.append(Y_train); Y_tests.append(Y_test)
    X_trainset = X_trains[0]; Y_trainset = Y_trains[0]; #X_trainset=X_trainset.fillna(0)
    # X_testset['add_idx_8']=0
    # import pdb;pdb.set_trace()
    if model == "svr":
        base = SVR()
    elif model == "rfr":
        base = RandomForestRegressor()
    # if print_idx == 0:
    #     X_test = X_testset
    #     y_test = Y_testset
    # else:
    #     X_test = X_tests[print_idx-1]
    #     y_test = Y_tests[print_idx-1]
    #     if print_idx>=1:
    #         X_test['add_idx_8']=0
    #     if print_idx>=4:
    #         X_test['add_idx_25']=0
    #     if print_idx>=5:
    #         X_test['add_idx_29']=0
    #     X_test=X_test[['age', 'CT_R', 'CT_E', 'dis', 'danger', 'rep_idx', 'cut_date',
    #    'gender_0', 'gender_1', 'is_korean_0', 'is_korean_1', 'primary case_0',
    #    'primary case_1', 'job_idx_0', 'job_idx_1', 'job_idx_2', 'job_idx_3',
    #    'job_idx_4', 'job_idx_5', 'job_idx_6', 'job_idx_7', 'job_idx_8',
    #    'job_idx_9', 'job_idx_99', 'place_idx_1', 'place_idx_2', 'place_idx_3',
    #    'place_idx_4', 'place_idx_5', 'place_idx_6', 'place_idx_7',
    #    'place_idx_8', 'place_idx_9', 'place_idx_10', 'place_idx_11',
    #    'place_idx_12', 'place_idx_13', 'place_idx_14', 'place_idx_15',
    #    'place_idx_16', 'place_idx_17', 'place_idx_18', 'place_idx_19',
    #    'add_idx_0', 'add_idx_1', 'add_idx_2', 'add_idx_3', 'add_idx_4',
    #    'add_idx_5', 'add_idx_6', 'add_idx_7', 'add_idx_8', 'add_idx_9',
    #    'add_idx_10', 'add_idx_11', 'add_idx_12', 'add_idx_13', 'add_idx_14',
    #    'add_idx_15', 'add_idx_16', 'add_idx_17', 'add_idx_18', 'add_idx_19',
    #    'add_idx_20', 'add_idx_21', 'add_idx_22', 'add_idx_23', 'add_idx_24',
    #    'add_idx_25', 'add_idx_26', 'add_idx_27', 'add_idx_28', 'add_idx_29',
    #    'add_idx_30']]
        # import pdb;pdb.set_trace()
    mo_model = MultiOutputRegressor(base)
    mo_model.fit(X_trainset, Y_trainset)

    # 예측 수행
    results = []
    y_pred_tr = mo_model.predict(X_trainset)
    for i in range(cutdates_num+1):
        y_pred = mo_model.predict(X_tests[i])
        y_pred_restored = sclaers[f'y_{i}'].inverse_transform(y_pred)[:,0]
        y_true_restored = sclaers[f'y_{i}'].inverse_transform(Y_tests[i].values)[:,0]
        d_pred_restored = sclaers[f'd_{i}'].inverse_transform(y_pred)[:,1]
        d_true_restored = sclaers[f'd_{i}'].inverse_transform(Y_tests[i].values)[:,1]
        # MAE, RMSE 평가 및 출력
        tr_mse = mean_squared_error(Y_trainset, y_pred_tr)
        mae_y = mean_absolute_error(y_true_restored, y_pred_restored)
        rmse_y = np.sqrt(mean_squared_error(y_true_restored, y_pred_restored))
    
        mae_d = mean_absolute_error(d_true_restored, d_pred_restored)
        rmse_d = np.sqrt(mean_squared_error(d_true_restored, d_pred_restored))
        
        date_key = "concat" if i == 0 else f"date_{i}"
        
        results.append([date_key, mae_d, mae_y, rmse_d, rmse_y])

        # Create a DataFrame to display the results
        columns = ["Date", "MAE (d)", "MAE (y)", "RMSE (d)", "RMSE (y)"]
        df_results = pd.DataFrame(results, columns=columns)
        if not ignore_wandb:
            wandb.log({"tr_loss" : tr_mse,
                    f"mae_loss (d) {date_key}" : mae_d,
                    f"mae_loss (y) {date_key}" : mae_y,
                    f"rmse_loss (d) {date_key}" : rmse_d,
                    f"rmse_loss (y) {date_key}" : rmse_y,
                    f"mae_loss {date_key}" : mae_d + mae_y,
                    f"rmse_loss {date_key}" : rmse_d + rmse_y,
                    })
    print(df_results)
    sys.exit()

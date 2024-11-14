import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import joblib

train_file_name = 'all_train.csv'
test_file_name = 'all_test.csv'


print('getting all_train')
df_train = pd.read_csv(train_file_name)
df_train = df_train.drop(['id',
'smpl'], axis=1)
x = df_train.drop(['target'], axis=1)
y = df_train['target']
print('all_train was got')


print('getting all_test')
df_test = pd.read_csv(test_file_name)
df_test_id = list(df_test['id'])
x_test = df_test.drop(['id','smpl'], axis=1)
print('all_test was got')


def model_create(params):
    print(f'model with params={params}')
    xgb_model = XGBClassifier(
        **params,
        eval_metric='auc',
        random_state=42,
    )

    xgb_model.fit(x, y, verbose=True)

    ans = xgb_model.predict_proba(x_test)[:,1]
    print('model was fitted')
    return ans
    
ans_1 = model_create(
    {
        'learning_rate': 0.01001912164665716,
        'n_estimators': 1999,
        'max_depth': 10,
        'min_child_weight': 10,
        'subsample': 0.8846534640278054,
        'colsample_bytree': 0.5001546567681449,
        'gamma': 4.5137567970035875,
        'reg_alpha': 0.31365093135737204,
        'reg_lambda': 4.139457827058301
    }
)

ans_2 = model_create(
    {
        'learning_rate': 0.012064622958533276,
        'n_estimators': 1070,
        'max_depth': 11,
        'min_child_weight': 8,
        'subsample': 0.8329492173246715,
        'colsample_bytree': 0.5165249730353096,
        'gamma': 3.781547794471798,
        'reg_alpha': 3.273808000793207,
        'reg_lambda': 2.4065599778221305
    }
)

ans_3 = model_create(
    {
        'learning_rate': 0.017190091885694803,
        'n_estimators': 875,
        'max_depth': 10,
        'min_child_weight': 19,
        'subsample': 0.816037626863144,
        'colsample_bytree': 0.5431077026379528,
        'gamma': 4.336681569568158,
        'reg_alpha': 1.8625910688144203,
        'reg_lambda': 3.3170357341294663
    }
)

ans_4 = model_create(
    {
        'learning_rate': 0.010547580366350617,
        'n_estimators': 973,
        'max_depth': 12,
        'min_child_weight': 16,
        'subsample': 0.9088148443706623,
        'colsample_bytree': 0.5432041890122357,
        'gamma': 2.6859045014707514,
        'reg_alpha': 0.8069781573144056,
        'reg_lambda': 3.174745132774029
    }
)

ans_5 = model_create(
    {
        'colsample_bytree': 0.5123444773635457,
        'gamma': 3.2047046133441905,
        'learning_rate': 0.010732141665630857,
        'max_depth': 12,
        'min_child_weight': 7,
        'n_estimators': 1663,
        'reg_alpha': 2.2054211366679644,
        'reg_lambda': 4.354291692182805,
        'subsample': 0.9023685900627255
    }
)

ans_6 = model_create(
    {
        'learning_rate': 0.010025022388458411,
        'n_estimators': 1118,
        'max_depth': 10,
        'min_child_weight': 7,
        'subsample': 0.8460813138757706,
        'colsample_bytree': 0.5447770782160501,
        'gamma': 4.580001004304903,
        'reg_alpha': 0.8799229904631172,
        'reg_lambda': 1.3400337507621933
    }
)

ans = []
for i in range(len(ans_1)):
    ans.append([
        df_test_id[i],
        (ans_1[i] + ans_2[i] + ans_3[i] + ans_4[i] + ans_5[i] + ans_6[i]) / 6
    ])

df_ans = pd.DataFrame(ans, columns=['id', 'target'])
df_ans.to_csv('predict.csv', index=False)
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
import os
import pickle
import pandas as pd

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
MODEL_ROOT = os.path.join(BASE_ROOT, 'models')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

BIAS_TERM_DICT = {
    'gender': ['male', 'female'], 
    'age': ['age_below_24', 'age_between_25_44', 'age_above_45'],
    'ethnicity': ['white', 'black', 'hispanic', 'other_races']
}

print('== Summary of Debiased Dataset ==')
debiased_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '04_debiased.csv')
debiased_df = pd.read_csv(debiased_path, low_memory=False)
print('Shape:', debiased_df.shape)
print('Unique years:', debiased_df['year'].nunique())
print('Unique ZIP codes:', debiased_df['zip_code'].nunique())

print('== Setup ==')
last_year = int(debiased_df['year'].max())
zip_code_list = list[int](debiased_df['zip_code'].unique())
identity_columns = pd.Index(['year', 'zip_code'])
majority_columns = pd.Index([column for column in debiased_df.columns if column.endswith('_majority')])
data_columns = debiased_df.columns.difference(identity_columns.union(majority_columns))

print('== Setup ==')
last_year = int(debiased_df['year'].max())
zip_code_list = list[int](debiased_df['zip_code'].unique())
total_lags = 1
max_depth = 9
n_estimators = 203
learning_rate = 0.001499

print('== Pipeline Retrieval ==')
best_pipeline_path = os.path.join(MODEL_ROOT, f'xgboost_{total_lags}_{max_depth}_{n_estimators}_{learning_rate}.pickle')
with open(best_pipeline_path, 'rb') as pipeline_file:
    scaler, model = tuple[StandardScaler, XGBClassifier](pickle.load(pipeline_file))

print('== Dataset Preparation ==')
zip_code_df_list = list[pd.DataFrame]()
for zip_code in zip_code_list:
    zip_code_df = debiased_df[debiased_df['zip_code'] == zip_code][identity_columns.union(data_columns)].copy()
    for column in data_columns:
        for lag in range(1, total_lags + 1):
            zip_code_df[f'{column}_lag_{lag}'] = zip_code_df[column].shift(lag)
    zip_code_df = zip_code_df.dropna().rename(columns={column: f'{column}_lag_0' for column in data_columns})
    zip_code_df[majority_columns] = debiased_df[debiased_df['zip_code'] == zip_code][majority_columns].iloc[-zip_code_df.shape[0]:]
    zip_code_df['target'] = zip_code_df['homeless_individuals_count_lag_0'].diff().gt(0).astype(int)
    zip_code_df = zip_code_df[zip_code_df['year'] == last_year]
    zip_code_df_list.append(zip_code_df.dropna(subset=['target']))
test_df = pd.concat(zip_code_df_list).sort_values(['year', 'zip_code']).reset_index(drop=True)

print('== Prediction ==')
feature_columns = pd.Index([f'{column}_lag_{lag}' for column in data_columns for lag in range(total_lags + 1)])
X_test = scaler.transform(test_df[feature_columns])
y_test = test_df['target'].values
y_test_pred = model.predict(X_test)

print('== Metrics ==')
test_f1_score = fbeta_score(y_test, y_test_pred, beta=1, average='binary')
test_f2_score = fbeta_score(y_test, y_test_pred, beta=2, average='binary')
print(f'Test F1 Score: {100 * test_f1_score:.4f}%, Test F2 Score: {100 * test_f2_score:.4f}%')

print('== Group Metrics (Before Debiasing) ==')
for group_name, group_list in BIAS_TERM_DICT.items():
    print(f'{group_name}:')
    for term in group_list:
        group_mask = test_df[f'{group_name}_majority'] == term
        group_f1_score = fbeta_score(test_df.loc[group_mask, 'target'], y_test_pred[group_mask], beta=1, average='binary')
        group_f2_score = fbeta_score(test_df.loc[group_mask, 'target'], y_test_pred[group_mask], beta=2, average='binary')
        print(f'  {term}: F1 Score: {100 * group_f1_score:.4f}%, F2 Score: {100 * group_f2_score:.4f}%')


print('== Debiasing ==')
for group_name, group_list in BIAS_TERM_DICT.items():
    for term in group_list:
        group_mask = test_df[f'{group_name}_majority'] == term
        y_prob = model.predict_proba(X_test[group_mask])[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test[group_mask], y_prob)
        f2_scores = 5 * (precisions * recalls) / (4 * precisions + recalls)
        best_threshold = float(thresholds[f2_scores.argmax()])
        y_test_pred[group_mask] = (y_prob >= best_threshold).astype(int)

print('== Group Metrics (Before Debiasing) ==')
for group_name, group_list in BIAS_TERM_DICT.items():
    print(f'{group_name}:')
    for term in group_list:
        group_mask = test_df[f'{group_name}_majority'] == term
        group_f1_score = fbeta_score(test_df.loc[group_mask, 'target'], y_test_pred[group_mask], beta=1, average='binary')
        group_f2_score = fbeta_score(test_df.loc[group_mask, 'target'], y_test_pred[group_mask], beta=2, average='binary')
        print(f'  {term}: F1 Score: {100 * group_f1_score:.4f}%, F2 Score: {100 * group_f2_score:.4f}%')
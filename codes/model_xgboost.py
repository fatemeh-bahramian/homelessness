from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
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

print('== Hyperparameter Search Space ==')
space_dict = [
    Integer(0, 3, name='total_lags'),
    Integer(6, 9, name='max_depth'),
    Integer(180, 220, name='n_estimators'),
    Real(0.0005, 0.0015, name='learning_rate')
]

print('== Bayesian Hyperparameter Search ==')
params_dict = dict[tuple[int, int, int, float], tuple[StandardScaler, XGBClassifier, float, float, float, float]]()
def objective(params: tuple[int, int, int, float]) -> float:

    # Getting the hyperparameters
    total_lags, max_depth, n_estimators, learning_rate = params
    print(f'Trial {len(params_dict) + 1} -> total_lags: {total_lags}, max_depth: {max_depth}, n_estimators: {n_estimators}, learning_rate: {learning_rate:.6f}')

    # Preparing the dataset
    zip_code_df_list = list[pd.DataFrame]()
    for zip_code in zip_code_list:
        zip_code_df = debiased_df[debiased_df['zip_code'] == zip_code][identity_columns.union(data_columns)].copy()
        for column in data_columns:
            for lag in range(1, total_lags + 1):
                zip_code_df[f'{column}_lag_{lag}'] = zip_code_df[column].shift(lag)
        zip_code_df = zip_code_df.dropna().rename(columns={column: f'{column}_lag_0' for column in data_columns})
        zip_code_df['target'] = zip_code_df['homeless_individuals_count_lag_0'].diff().gt(0).astype(int)
        zip_code_df_list.append(zip_code_df.dropna(subset=['target']))
    dev_df = pd.concat(zip_code_df_list).sort_values(['year', 'zip_code']).reset_index(drop=True)

    # Making Train/Test Splits
    train_df = dev_df[dev_df['year'] < last_year]
    test_df = dev_df[dev_df['year'] == last_year]

    # Standardizing Features
    feature_columns = pd.Index([f'{column}_lag_{lag}' for column in data_columns for lag in range(total_lags + 1)])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])
    y_train = train_df['target'].values
    y_test = test_df['target'].values

    # Train model (no obsolete params)
    model = XGBClassifier(eval_metric='logloss', max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train, verbose=True)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_f1_score = fbeta_score(y_train, y_train_pred, beta=1, average='binary')
    train_f2_score = fbeta_score(y_train, y_train_pred, beta=2, average='binary')
    test_f1_score = fbeta_score(y_test, y_test_pred, beta=1, average='binary')
    test_f2_score = fbeta_score(y_test, y_test_pred, beta=2, average='binary')
    print(f'  train_f1_score: {100 * train_f1_score:.4f}%, train_f2_score: {100 * train_f2_score:.4f}%, test_f1_score: {100 * test_f1_score:.4f}%, test_f2_score: {100 * test_f2_score:.4f}%')

    # Return
    params_dict[tuple(params)] = (scaler, model, train_f1_score, train_f2_score, test_f1_score, test_f2_score)
    return -(test_f1_score + test_f2_score)

search_result = gp_minimize(
    func=objective,
    dimensions=space_dict,
    acq_func='EI',      # Expected Improvement
    n_calls=60,
    n_random_starts=5,
    random_state=42
)

print('== Best Hyperparameters ==')
best_params = tuple(search_result.x)
best_total_lags, best_max_depth, best_n_estimators, best_learning_rate = best_params
best_scaler, best_model, best_train_f1_score, best_train_f2_score, best_test_f1_score, best_test_f2_score = params_dict[best_params]
print(f'Best total_lags: {best_total_lags}')
print(f'Best max_depth: {best_max_depth}')
print(f'Best n_estimators: {best_n_estimators}')
print(f'Best learning_rate: {best_learning_rate:.6f}')
print(f'Best train_f1_score: {100 * best_train_f1_score:.4f}%')
print(f'Best train_f2_score: {100 * best_train_f2_score:.4f}%')
print(f'Best test_f1_score: {100 * best_test_f1_score:.4f}%')
print(f'Best test_f2_score: {100 * best_test_f2_score:.4f}%')

print('== Pipeline Storage ==')
best_pipeline_path = os.path.join(MODEL_ROOT, f'xgboost_{best_total_lags}_{best_max_depth}_{best_n_estimators}_{best_learning_rate:.6f}.pickle')
with open(best_pipeline_path, 'wb') as pipeline_file:
    pickle.dump((best_scaler, best_model), pipeline_file)

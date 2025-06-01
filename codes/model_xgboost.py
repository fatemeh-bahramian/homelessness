from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
import os
import pickle
import pandas as pd

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
results_dict = dict[tuple[int, int, int, float], tuple[StandardScaler, XGBClassifier, tuple[float, float, float]]]()
def objective(params: tuple[int, int, int, float]) -> float:

    # Getting the hyperparameters
    total_lags, max_depth, n_estimators, learning_rate = params
    print(f'Trial {len(results_dict) + 1} -> total_lags: {total_lags}, max_depth: {max_depth}, n_estimators: {n_estimators}, learning_rate: {learning_rate:.6f}')

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
    train_f_scores = tuple(fbeta_score(y_train, y_train_pred, beta=beta, average='binary') for beta in [0.5, 1, 2])
    test_f_scores = tuple(fbeta_score(y_test, y_test_pred, beta=beta, average='binary') for beta in [0.5, 1, 2])
    print(f'  Train -> F0.5-Score {100 * train_f_scores[0]:.4f}%, F1-Score: {100 * train_f_scores[1]:.4f}%, F2-Score: {100 * train_f_scores[2]:.4f}%')
    print(f'  Test  -> F0.5-Score {100 * test_f_scores[0]:.4f}%, F1-Score: {100 * test_f_scores[1]:.4f}%, F2-Score: {100 * test_f_scores[2]:.4f}%')

    # Return
    results_dict[tuple(params)] = (scaler, model, test_f_scores)
    return -sum(test_f_scores)

search_result = gp_minimize(
    func=objective,
    dimensions=space_dict,
    acq_func='EI',      # Expected Improvement
    n_calls=60,
    n_random_starts=5,
    random_state=42
)

print('== Best Hyperparameters ==')
best_f_score_list = [0] * 3
best_f_scores_list = list[tuple[float, float, float]]([(None, None, None)] * 3)
best_params_list = list[tuple[int, int, int, float]]([(None, None, None, None)] * 3)
best_pipeline_list = list[tuple[StandardScaler, XGBClassifier]]([None] * 3)
for params in results_dict:
    scaler, model, test_f_scores = results_dict[params]
    for k in range(3):
        if test_f_scores[k] > best_f_score_list[k]:
            best_f_score_list[k] = test_f_scores[k]
            best_f_scores_list[k] = test_f_scores
            best_params_list[k] = params
            best_pipeline_list[k] = (scaler, model)
for k, beta in enumerate(['0_5', '1', '2']):
    print(f'Regarding F{beta}-Score:')
    print(f'  Best Hyperparameters -> total_lags: {best_params_list[k][0]}, max_depth: {best_params_list[k][1]}, n_estimators: {best_params_list[k][2]}, learning_rate: {best_params_list[k][3]:.6f}')
    print(f'  Best F-Scores: F0.5-Score {100 * best_f_scores_list[k][0]:.4f}%, F1-Score: {100 * best_f_scores_list[k][1]:.4f}%, F2-Score: {100 * best_f_scores_list[k][2]:.4f}%')

print('== Pipeline Storage ==')
for k, beta in enumerate(['0_5', '1', '2']):
    best_scaler, best_model = best_pipeline_list[k]
    pipeline_path = os.path.join(MODEL_ROOT, f'xgboost_f{beta}_score.pickle')
    with open(pipeline_path, 'wb') as pipeline_file:
        pickle.dump((best_scaler, best_model), pipeline_file)
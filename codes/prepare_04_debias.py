import os
import pandas as pd

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

BIAS_TERM_DICT = {
    'gender': ['male', 'female'], 
    'age': ['age_below_24', 'age_between_25_44', 'age_above_45'],
    'ethnicity': ['white', 'black', 'hispanic', 'other_races']
}

print('== Summary of Cleaned Dataset ==')
cleaned_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '03_cleaned.csv')
cleaned_df = pd.read_csv(cleaned_path, low_memory=False)
print('Shape:', cleaned_df.shape)
print('Total unique Years:', cleaned_df['year'].nunique())
print('Total unique ZIP Codes:', cleaned_df['zip_code'].nunique())

print('== Debiasing ==')
debiased_df = cleaned_df.copy()
for column in cleaned_df.columns:
    is_biased = False
    for bias_term in [term for term_list in BIAS_TERM_DICT.values() for term in term_list]:
        if column.endswith(bias_term):
            is_biased = True
            break
    if is_biased:
        debiased_df.drop(columns=[column], inplace=True)
print('Shape:', debiased_df.shape)
print('Total unique Years:', debiased_df['year'].nunique())
print('Total unique ZIP Codes:', debiased_df['zip_code'].nunique())

groups_population_df = cleaned_df \
    .drop(columns=[column for column in cleaned_df.columns if not column.startswith('population_')]) \
    .assign(zip_code=cleaned_df['zip_code']).reset_index(drop=True)
groups_population_df.columns = groups_population_df.columns.str.replace('population_', '')
for group_name, group_list in BIAS_TERM_DICT.items():
    debiased_df[f'{group_name}_majority'] = groups_population_df[group_list].idxmax(axis=1)

print('== Storage ==')
debiased_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '04_debiased.csv')
debiased_df.to_csv(debiased_path, index=False)

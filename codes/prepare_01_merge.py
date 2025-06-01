import os
import pandas as pd

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

transformed_file_name_list = list[str]()
for transformed_file_name in os.listdir(TRANSFORMED_DATA_FOLDER_PATH):
    if transformed_file_name.endswith('.csv'):
        transformed_file_name_list.append(transformed_file_name)
transformed_file_name_list.sort()

year_set = set[int]()
zip_code_set = set[int]()
transformed_df_list = list[pd.DataFrame]()
for transformed_file_name in transformed_file_name_list:

    print(f'== {transformed_file_name} ==')

    print('  Loading ...')
    transformed_path = os.path.join(TRANSFORMED_DATA_FOLDER_PATH, transformed_file_name)
    transformed_df = pd.read_csv(transformed_path, low_memory=False)

    print('  Minor Processing ...')
    year_set.update(transformed_df['year'].unique().tolist())
    zip_code_set.update(transformed_df['zip_code'].unique().tolist())
    transformed_df_list.append(transformed_df)

print('== Merging ==')
merged_df = pd.MultiIndex.from_product([sorted(year_set), sorted(zip_code_set)], names=['year', 'zip_code']).to_frame(index=False)
for transformed_df in transformed_df_list:
    merged_df = pd.merge(merged_df, transformed_df, on=['year', 'zip_code'], how='outer')
merged_df = merged_df.sort_values(['year', 'zip_code']).reset_index(drop=True)

print('== Storage ==')
merged_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '01_merged.csv')
merged_df.to_csv(merged_path, index=False)
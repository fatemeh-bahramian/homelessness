import os
import pandas as pd

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')
MERGED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'merged')

transformed_file_name_list = list[str]()
for transformed_file_name in os.listdir(TRANSFORMED_DATA_FOLDER_PATH):
    if transformed_file_name.endswith('.csv'):
        transformed_file_name_list.append(transformed_file_name)
transformed_file_name_list.sort()

for transformed_file_name in transformed_file_name_list:

    print(f'== {transformed_file_name} ==')
    transformed_path = os.path.join(TRANSFORMED_DATA_FOLDER_PATH, transformed_file_name)
    transformed_df = pd.read_csv(transformed_path, low_memory=False)

    unique_year_zip_code_count_sr = transformed_df.apply(lambda x: str(int(x['year'])) + '_' + str(int(x['zip_code'])), axis=1).value_counts()
    print('  total duplicates', (unique_year_zip_code_count_sr > 1).sum())

    total_unique_years = transformed_df['year'].nunique()
    total_unique_zip_codes = transformed_df['zip_code'].nunique()
    print('  total unique years:', total_unique_years)
    print('  total unique zip codes:', total_unique_zip_codes)

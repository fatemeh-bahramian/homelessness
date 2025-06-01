import os
import random
import pandas as pd
import numpy as np

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

print('== Summary of Filtered Dataset ==')
filtered_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '02_filtered.csv')
filtered_df = pd.read_csv(filtered_path, low_memory=False)
print('Shape:', filtered_df.shape)
print('Total unique years:', filtered_df['year'].nunique())
print('Total unique ZIP codes:', filtered_df['zip_code'].nunique())

print('== Setup ==')
zip_code_list = list[int](filtered_df['zip_code'].unique())

print('== Cleaning Income Columns ==')
income_column_list = [column for column in filtered_df.columns if column.startswith('median_income')]
for zip_code in zip_code_list:
    zip_code_mask = filtered_df['zip_code'] == zip_code
    for column in income_column_list:
        income_mask = filtered_df.loc[zip_code_mask, column] < filtered_df.loc[zip_code_mask, column].mean()
        filtered_df.loc[zip_code_mask & income_mask, column] *= 1000.0
print('Shape:', filtered_df.shape)
print('Total unique Years:', filtered_df['year'].nunique())
print('Total unique ZIP Codes:', filtered_df['zip_code'].nunique())

print('== Cleaning All Columns ==')
data_columns = filtered_df.columns.difference(['year', 'zip_code'])
zip_code_df_list = list[pd.DataFrame]()
for zip_code in zip_code_list:
    zip_code_df = filtered_df[filtered_df['zip_code'] == zip_code].copy()
    zip_code_df[zip_code_df == 0.0] = np.nan
    zip_code_df[data_columns] = zip_code_df[data_columns].interpolate(method='linear', limit_direction='both', axis=0).bfill().ffill().fillna(0.0)
    zip_code_df_list.append(zip_code_df)
clean_df = pd.concat(zip_code_df_list).sort_values(['year', 'zip_code']).reset_index(drop=True)
print('Shape:', clean_df.shape)
print('Total unique Years:', clean_df['year'].nunique())
print('Total unique ZIP Codes:', clean_df['zip_code'].nunique())

print('== Storage ==')
clean_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '03_cleaned.csv')
clean_df.to_csv(clean_path, index=False)
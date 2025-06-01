import os
import pandas as pd

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

print('== Summary of Merged Dataset ==')
merged_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '01_merged.csv')
merged_df = pd.read_csv(merged_path, low_memory=False)
print('Shape:', merged_df.shape)
print('Total unique years:', merged_df['year'].nunique())
print('Total unique ZIP codes:', merged_df['zip_code'].nunique())

print('== Filtering ==')
years_with_minimum_non_null_columns:list[int] = merged_df.groupby('year') \
    .apply(lambda column_sr: column_sr.isnull().all(), include_groups=False) \
    .sum(axis=1).apply(lambda total_null_columns: total_null_columns / (merged_df.shape[1] - 1)) \
    .pipe(lambda sr: sr[sr < 0.1]).index.tolist()
zip_code_with_non_null_columns_list:list[int] = merged_df.groupby('zip_code') \
    .apply(lambda column_sr: column_sr.isnull().all(), include_groups=False) \
    .any(axis=1).pipe(lambda sr: sr[~sr].index.tolist())
filtered_df = merged_df[merged_df['zip_code'].isin(zip_code_with_non_null_columns_list)]\
    .sort_values(['year', 'zip_code']).reset_index(drop=True)
filtered_df = filtered_df[
    (min(years_with_minimum_non_null_columns) <= filtered_df['year']) & \
    (filtered_df['year'] <= max(years_with_minimum_non_null_columns))\
].reset_index(drop=True)
print('Shape:', filtered_df.shape)
print('Total unique ZIP codes with non-null columns:', len(zip_code_with_non_null_columns_list))
print('Total unique years with minimum non-null columns:', len(years_with_minimum_non_null_columns))

print('== Storage ==')
filtered_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '02_filtered.csv')
filtered_df.to_csv(filtered_path, index=False)
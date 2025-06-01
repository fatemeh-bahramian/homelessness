import os
import pandas as pd

DATASET_NAME = os.path.basename(__file__)[10:-3]

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
CROSS_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', 'crosswalk', 'csv')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME, 'csv')
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')

cross_path = os.path.join(CROSS_DATA_FOLDER_PATH, 'crosswalk.csv')
cross_df = pd.read_csv(cross_path, low_memory=False, dtype=str)

raw_file_name_list = list[str]()
for raw_file_name in os.listdir(RAW_DATA_FOLDER_PATH):
    if raw_file_name.endswith('.csv'):
        raw_file_name_list.append(raw_file_name)
raw_file_name_list.sort()

agg_df_list = list[pd.DataFrame]()
for raw_file_name in raw_file_name_list:

    print(f'== {raw_file_name} ==')

    print('  Loading ...')
    raw_path = os.path.join(RAW_DATA_FOLDER_PATH, raw_file_name)
    raw_df = pd.read_csv(raw_path, low_memory=False, dtype=str)
    raw_df['tract'] = '06037' + raw_df['tract'].str.zfill(6)

    print('  Crosswalk ...')
    quarter_list = cross_df['quarter'].unique().tolist()
    max_tracts_covered = 0
    max_quarter = None
    max_cross_df = None
    for quarter in quarter_list:
        temp_cross_df:pd.DataFrame = cross_df[cross_df['quarter'] == quarter]
        total_tracts_covered = raw_df['tract'].isin(temp_cross_df['tract']).sum()
        if total_tracts_covered > max_tracts_covered:
            max_tracts_covered = total_tracts_covered
            max_quarter = quarter
            max_cross_df = temp_cross_df
    print('    Maximal Quarter:', max_quarter)

    print('  Transformation ...')
    zip_code_list = max_cross_df['zip_code'].unique().tolist()
    agg_row_list = list[dict[str, str|pd.Series]]()
    for zip_code in zip_code_list:
        tract_ts = max_cross_df[max_cross_df['zip_code'] == zip_code]['tract']
        row = {
            'year': raw_file_name[:4],
            'zip_code': zip_code,
            'cars_count': pd.to_numeric(raw_df[raw_df['tract'].isin(tract_ts)]['total cars'], errors='coerce').sum(min_count=1),
            'vans_count': pd.to_numeric(raw_df[raw_df['tract'].isin(tract_ts)]['total vans'], errors='coerce').sum(min_count=1),
            'campers_or_rvs_count': pd.to_numeric(raw_df[raw_df['tract'].isin(tract_ts)]['total campers or rvs'], errors='coerce').sum(min_count=1),
            'tents_count': pd.to_numeric(raw_df[raw_df['tract'].isin(tract_ts)]['total tents'], errors='coerce').sum(min_count=1),
            'homeless_individuals_count': pd.to_numeric(raw_df[raw_df['tract'].isin(tract_ts)]['total homeless individuals'], errors='coerce').sum(min_count=1)
        }
        agg_row_list.append(row)
    agg_df = pd.DataFrame(agg_row_list)

    print('  Minor Processing ...')
    agg_df_list.append(agg_df)

print('== Storage ==')
transformed_df = pd.concat(agg_df_list, ignore_index=True)
transformed_path = os.path.join(TRANSFORMED_DATA_FOLDER_PATH, f'{DATASET_NAME}.csv')
transformed_df.to_csv(transformed_path, index=False)
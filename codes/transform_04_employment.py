import os
import pandas as pd
import geopandas as gpd

DATASET_NAME = os.path.basename(__file__)[10:-3]

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', 'zip')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')

GROUPING_DICT = {
    'unemployment_rate': ['S2301_C04_001E'],  # Total
    'unemployment_rate_male': ['S2301_C04_020E'],  # Male
    'unemployment_rate_female': ['S2301_C04_021E'], # Female
    'unemployment_rate_age_below_24': [
        'S2301_C04_002E',  # 16-19
        'S2301_C04_003E'   # 20-24
    ],
    'unemployment_rate_age_between_25_44': [
        'S2301_C04_004E'   # 25-44
    ],
    'unemployment_rate_age_above_45': [
        'S2301_C04_005E',  # 45-54
        'S2301_C04_006E',  # 55-64
        'S2301_C04_007E',  # 65-74
        'S2301_C04_008E'   # 75+
    ],
    'unemployment_rate_white': ['S2301_C04_010E'],  # White
    'unemployment_rate_black': ['S2301_C04_011E'],  # Black
    'unemployment_rate_hispanic': ['S2301_C04_017E'],  # Hispanic, Latin, or Mexican
    'unemployment_rate_other_races': [
        'S2301_C04_012E', # Native American or Alaskan Native
        'S2301_C04_013E', # Chinese, Japanese, Korean, Vietnamese, Asian Indian, Filipino, Other Asian
        'S2301_C04_014E', 'S2301_C02_014E',  # Native Hawaiian
        'S2301_C04_015E', 'S2301_C02_015E',  # Other race
        'S2301_C04_016E', 'S2301_C02_016E'   # Two or more races
    ]
}

zip_path = os.path.join(ZIP_DATA_FOLDER_PATH, 'City_of_Los_Angeles_Zip_Codes.shp')
zip_gdf: gpd.GeoDataFrame = gpd.read_file(zip_path)
zip_gdf['ZCTA5CE10'] = zip_gdf['ZCTA5CE10'].astype(str).str.zfill(5)

raw_file_name_list = list[str]()
for raw_file_name in os.listdir(RAW_DATA_FOLDER_PATH):
    if raw_file_name.endswith('Data.csv'):
        raw_file_name_list.append(raw_file_name)
raw_file_name_list.sort()

raw_df_list = list[pd.DataFrame]()
for raw_file_name in raw_file_name_list:

    print(f'== {raw_file_name} ==')

    print('  Loading ...')
    raw_path = os.path.join(RAW_DATA_FOLDER_PATH, raw_file_name)
    raw_df = pd.read_csv(raw_path, low_memory=False)

    print('  Local Filtering ...')
    raw_df['zip_code'] = raw_df['NAME'].str.extract(r'ZCTA5 (\d{5})')[0].astype(str).str.zfill(5)
    raw_df = raw_df[raw_df['zip_code'].isin(zip_gdf['ZCTA5CE10'])]
    for column_name in raw_df.columns:
        if column_name.endswith('E'):  # Only estimate columns
            raw_df[column_name] = pd.to_numeric(raw_df[column_name], errors='coerce').astype(float)

    print('  Minor Processing ...')
    raw_df = raw_df.assign(year=raw_file_name[7:11])
    raw_df_list.append(raw_df)

print('== Transformation ==')
raw_df = pd.concat(raw_df_list, ignore_index=True)
transformed_df = pd.DataFrame({
    'year': raw_df['year'],
    'zip_code': raw_df['zip_code']
})
for group_name, column_name_list in GROUPING_DICT.items():
    transformed_df[group_name] = raw_df[column_name_list].sum(axis=1, min_count=1)
transformed_df.sort_values(['year', 'zip_code'], inplace=True)
    
print('== Storage ==')
transformed_path = os.path.join(TRANSFORMED_DATA_FOLDER_PATH, f'{DATASET_NAME}.csv')
transformed_df.to_csv(transformed_path, index=False)
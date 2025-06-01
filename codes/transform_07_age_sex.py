import os
import pandas as pd
import geopandas as gpd

DATASET_NAME = os.path.basename(__file__)[10:-3]

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'utility', 'shape_files')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')

GROUPING_DICT = {
    'population': ['B01001_001E'],  # Total
    'population_male': ['B01001_002E'],  # Male
    'population_female': ['B01001_026E'],  # Female
    'population_age_below_24': [
        'B01001_003E', 'B01001_004E', 'B01001_005E', 'B01001_006E', 'B01001_007E', 'B01001_008E', 'B01001_009E', 'B01001_010E',  # Male: 0-5, 5-9, 10-14, 15-17, 18-19, 20, 21, 22-24
        'B01001_027E', 'B01001_028E', 'B01001_029E', 'B01001_030E', 'B01001_031E', 'B01001_032E', 'B01001_033E', 'B01001_034E'  # Female: 0-5, 5-9, 10-14, 15-17, 18-19, 20, 21, 22-24
    ],
    'population_age_between_25_44': [
        'B01001_011E', 'B01001_012E', 'B01001_013E', 'B01001_014E', # Male: 25-29, 30-34, 35-39, 40-44
        'B01001_035E', 'B01001_036E', 'B01001_037E', 'B01001_038E' # Female: 25-29, 30-34, 35-39, 40-44
    ],
    'population_age_above_45': [
        'B01001_015E', 'B01001_016E', 'B01001_017E', 'B01001_018E', 'B01001_019E', 'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E', # Male: 45-49, 50-54, 55-59, 60-61, 62-64, 65-66, 67-69, 70-74, 75-79, 80-84, 85+
        'B01001_039E', 'B01001_040E', 'B01001_041E', 'B01001_042E', 'B01001_043E', 'B01001_044E', 'B01001_045E', 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E' # Female: 45-49, 50-54, 55-59, 60-61, 62-64, 65-66, 67-69, 70-74, 75-79, 80-84, 85+
    ]
}

zip_path = os.path.join(ZIP_DATA_FOLDER_PATH, 'City_of_Los_Angeles_Zip_Codes.shp')
zip_gdf:gpd.GeoDataFrame = gpd.read_file(zip_path)
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
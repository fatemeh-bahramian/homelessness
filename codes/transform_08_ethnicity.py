import os
import geopandas as gpd
import pandas as pd

DATASET_NAME = os.path.basename(__file__)[10:-3]

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'utility', 'shape_files')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')

GROUPING_DICT = {
    'population_white': [
        'B03002_003E',  # Non-Hispanic White alone
        'B03002_013E'   # Hispanic White alone
    ],
    'population_black': [
        'B03002_004E',  # Non-Hispanic Black or African American alone
        'B03002_014E'   # Hispanic Black or African American alone
    ],
    'population_hispanic': [
        'B03002_012E'   # Total Hispanic or Latino population (any race)
    ],
    'population_other_races': [
        'B03002_005E',  # Non-Hispanic American Indian and Alaska Native alone
        'B03002_015E',  # Hispanic American Indian and Alaska Native alone
        'B03002_006E',  # Non-Hispanic Asian alone
        'B03002_016E',  # Hispanic Asian alone
        'B03002_007E',  # Non-Hispanic Native Hawaiian and Other Pacific Islander alone
        'B03002_017E',  # Hispanic Native Hawaiian and Other Pacific Islander alone
        'B03002_008E',  # Non-Hispanic Some Other Race alone
        'B03002_018E',  # Hispanic Some Other Race alone
        'B03002_009E',  # Non-Hispanic Two or More Races
        'B03002_019E'   # Hispanic Two or More Races
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
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
    'below_poverty_level_individuals_count': [
        'S1701_C02_001E'  # Total population
    ],
    'below_poverty_level_individuals_count_male': [
        'S1701_C02_006E'  # Male
    ],
    'below_poverty_level_individuals_count_female': [
        'S1701_C02_007E'  # Female
    ],
    'below_poverty_level_individuals_count_age_below_24': [
        'S1701_C02_002E'  # Under 18 years
    ],
    'below_poverty_level_individuals_count_age_between_25_44': [
        'S1701_C02_004E'  # 19-64 years (using as proxy with limitation)
    ],
    'below_poverty_level_individuals_count_age_above_45': [
        'S1701_C02_005E'  # 65 years and over
    ],
    'below_poverty_level_individuals_count_white': [
        'S1701_C02_009E'  # White
    ],
    'below_poverty_level_individuals_count_black': [
        'S1701_C02_010E'  # Black or African American
    ],
    'below_poverty_level_individuals_count_hispanic': [
        'S1701_C02_016E'  # Hispanic or Latino origin (of any race)
    ],
    'below_poverty_level_individuals_count_other_races': [
        'S1701_C02_011E',  # American Indian and Alaska Native
        'S1701_C02_012E',  # Asian
        'S1701_C02_013E',  # Native Hawaiian and Other Pacific Islander
        'S1701_C02_014E',  # Some other race
        'S1701_C02_015E'   # Two or more races
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
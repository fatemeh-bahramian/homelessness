import os
import geopandas as gpd
import pandas as pd

DATASET_NAME = os.path.basename(__file__)[10:-3]

BASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', 'zip')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')

GROUPING_DICT = {
    'median_income': [
        'S1903_C02_001E',  # Median income (dollars)!!Estimate!!Households
    ],
    'median_income_male': [
        'S1903_C02_020E',  # Median income (dollars)!!Estimate!!FAMILIES!!Male householder, no wife present
        'S1903_C02_025E',  # Median income (dollars)!!Estimate!!NONFAMILY HOUSEHOLDS!!Male householder
        'S1903_C02_026E',  # Median income (dollars)!!Estimate!!NONFAMILY HOUSEHOLDS!!Male householder!!Living alone
        'S1903_C02_027E',  # Median income (dollars)!!Estimate!!NONFAMILY HOUSEHOLDS!!Male householder!!Not living alone
    ],
    'median_income_female': [
        'S1903_C02_019E',  # Median income (dollars)!!Estimate!!FAMILIES!!Female householder, no husband present
        'S1903_C02_022E',  # Median income (dollars)!!Estimate!!NONFAMILY HOUSEHOLDS!!Female householder
        'S1903_C02_023E',  # Median income (dollars)!!Estimate!!NONFAMILY HOUSEHOLDS!!Female householder!!Living alone
        'S1903_C02_024E',  # Median income (dollars)!!Estimate!!NONFAMILY HOUSEHOLDS!!Female householder!!Not living alone
    ],
    'median_income_age_below_24': [
        'S1903_C02_016E',  # Median income (dollars)!!Estimate!!FAMILIES!!Families!!With own children under 18 years
        'S1903_C02_011E',  # Median income (dollars)!!Estimate!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!15 to 24 years
    ],
    'median_income_age_between_25_44': [
        'S1903_C02_012E',  # Median income (dollars)!!Estimate!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!25 to 44 years
    ],
    'median_income_age_above_45': [
        'S1903_C02_013E',  # Median income (dollars)!!Estimate!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!45 to 64 years
        'S1903_C02_014E',  # Median income (dollars)!!Estimate!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!65 years and over
    ],
    'median_income_white': [
        'S1903_C02_002E',  # Median income (dollars)!!Estimate!!One race!!White
        'S1903_C02_010E',  # Median income (dollars)!!Estimate!!White alone, not Hispanic or Latino
    ],
    'median_income_black': [
        'S1903_C02_003E',  # Median income (dollars)!!Estimate!!One race!!Black or African American
    ],
    'median_income_hispanic': [
        'S1903_C02_009E',  # Median income (dollars)!!Estimate!!Hispanic or Latino origin (of any race)
    ],
    'median_income_other_races': [
        'S1903_C02_004E',  # Median income (dollars)!!Estimate!!One race!!American Indian and Alaska Native
        'S1903_C02_005E',  # Median income (dollars)!!Estimate!!One race!!Asian
        'S1903_C02_006E',  # Median income (dollars)!!Estimate!!One race!!Native Hawaiian and Other Pacific Islander
        'S1903_C02_007E',  # Median income (dollars)!!Estimate!!One race!!Some other race
        'S1903_C02_008E',  # Median income (dollars)!!Estimate!!Two or more races
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
    transformed_df[group_name] = raw_df[column_name_list].mean(axis=1, skipna=True)
transformed_df.sort_values(['year', 'zip_code'], inplace=True)

print('== Storage ==')
transformed_path = os.path.join(TRANSFORMED_DATA_FOLDER_PATH, f'{DATASET_NAME}.csv')
transformed_df.to_csv(transformed_path, index=False)
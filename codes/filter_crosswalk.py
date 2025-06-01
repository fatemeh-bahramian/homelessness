import os
import geopandas as gpd
import pandas as pd

DATASET_NAME = os.path.basename(__file__)[7:-3]

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', 'zip')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)
CLEAN_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME, 'csv')

zip_path = os.path.join(ZIP_DATA_FOLDER_PATH, 'City_of_Los_Angeles_Zip_Codes.shp')
zip_gdf:gpd.GeoDataFrame = gpd.read_file(zip_path)
zip_gdf['ZCTA5CE10'] = zip_gdf['ZCTA5CE10'].astype(str).str.zfill(5)

raw_file_name_list = list[str]()
for raw_file_name in os.listdir(RAW_DATA_FOLDER_PATH):
    if raw_file_name.endswith('.xlsx'):
        raw_file_name_list.append(raw_file_name)
raw_file_name_list.sort()

raw_df_list = list[pd.DataFrame]()
for raw_file_name in raw_file_name_list:

    print(f'== {raw_file_name} ==')

    print('  Loading ...')
    raw_path = os.path.join(RAW_DATA_FOLDER_PATH, raw_file_name)
    raw_df = pd.read_excel(raw_path, engine='openpyxl', sheet_name=0, dtype=str)
    zip_idx = next((i for i, col in enumerate(raw_df.columns.str.lower()) if 'zip' in col), None)
    tract_idx = next((i for i, col in enumerate(raw_df.columns.str.lower()) if 'tract' in col), None)
    raw_df:pd.DataFrame = raw_df.iloc[:, [zip_idx, tract_idx]]
    raw_df.columns = ['zip_code', 'tract']

    print('  Cleaning ...')
    raw_df['quarter'] = raw_file_name[-9:-5] + '_' + raw_file_name[-11:-9]
    raw_df['zip_code'] = raw_df['zip_code'].str.zfill(5)
    raw_df = raw_df[raw_df['zip_code'].isin(zip_gdf['ZCTA5CE10'])]
    raw_df = raw_df[['quarter'] + [col for col in raw_df.columns if col != 'quarter']]

    raw_df_list.append(raw_df)

clean_df = pd.concat(raw_df_list, ignore_index=True)
clean_path = os.path.join(CLEAN_DATA_FOLDER_PATH, f'{DATASET_NAME}.csv')
if not os.path.exists(CLEAN_DATA_FOLDER_PATH):
    os.makedirs(CLEAN_DATA_FOLDER_PATH)
clean_df.to_csv(clean_path, index=False)
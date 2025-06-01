from shapely.geometry import Point
import os
import pandas as pd
import geopandas as gpd

DATASET_NAME = os.path.basename(__file__)[17:-3]

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', 'zip')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)
TRANSFORMED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'transformed')

GROUPING_DICT = {
    'W': 'victims_count_white',  # White
    'B': 'victims_count_black',  # Black
    'H': 'victims_count_hispanic',  # Hispanic, Latin, or Mexican
    'C': 'victims_count_other_races',  # Chinese
    'J': 'victims_count_other_races',  # Japanese
    'K': 'victims_count_other_races',  # Korean
    'V': 'victims_count_other_races',  # Vietnamese
    'Z': 'victims_count_other_races',  # Asian Indian
    'F': 'victims_count_other_races',  # Filipino (grouped as Asian here)
    'A': 'victims_count_other_races',  # Other Asian (assumed East Asian)
    'I': 'victims_count_other_races',  # American Indian/Alaskan Native
    'D': 'victims_count_other_races',  # Cambodian (Southeast Asian)
    'G': 'victims_count_other_races',  # Guamanian (Pacific Islander)
    'L': 'victims_count_other_races',  # Laotian (Southeast Asian)
    'P': 'victims_count_other_races',  # Pacific Islander
    'S': 'victims_count_other_races',  # Samoan (Pacific Islander)
    'U': 'victims_count_other_races',  # Hawaiian (Pacific Islander)
    'O': 'victims_count_other_races',  # Other
    'X': 'victims_count_other_races',  # Unknown
    '-': 'victims_count_other_races',  # Unknown
    '': 'victims_count_other_races'   # Unknown
}

zip_path = os.path.join(ZIP_DATA_FOLDER_PATH, 'City_of_Los_Angeles_Zip_Codes.shp')
zip_gdf:gpd.GeoDataFrame = gpd.read_file(zip_path)
zip_gdf['ZCTA5CE10'] = zip_gdf['ZCTA5CE10'].astype(str).str.zfill(5)
zip_gdf = zip_gdf.to_crs(epsg=4326)

raw_file_name_list = list[str]()
for raw_file_name in os.listdir(RAW_DATA_FOLDER_PATH):
    if raw_file_name.endswith('.csv'):
        raw_file_name_list.append(raw_file_name)
raw_file_name_list.sort()

raw_df_list = list[pd.DataFrame]()
for raw_file_name in raw_file_name_list:

    print(f'== {raw_file_name} ==')

    print('  Loading ...')
    raw_path = os.path.join(RAW_DATA_FOLDER_PATH, raw_file_name)
    raw_df = pd.read_csv(raw_path, low_memory=False)

    print('  Local Filtering ...')
    raw_df = raw_df[(raw_df['LAT'] != 0.0) & (raw_df['LON'] != 0.0)]
    raw_df = raw_df[raw_df['Date Rptd'].notna()]

    print('  Geo-Processing ...')
    raw_gdf = gpd.GeoDataFrame(raw_df, geometry=[Point(xy) for xy in zip(raw_df['LON'], raw_df['LAT'])], crs='EPSG:4326')
    raw_gdf:gpd.GeoDataFrame = gpd.sjoin(raw_gdf, zip_gdf[['ZCTA5CE10', 'geometry']], how='left', predicate='within')
    
    print('  Minor Processing ...')
    raw_df = pd.DataFrame(raw_gdf.drop(columns='geometry'))
    raw_df['Date Rptd'] = pd.to_datetime(raw_df['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    raw_df['year'] = raw_df['Date Rptd'].dt.year
    raw_df = raw_df.rename(columns={'ZCTA5CE10': 'zip_code'})
    raw_df_list.append(raw_df)


print('== Transformation ==')
raw_df = pd.concat(raw_df_list, ignore_index=True)
transformed_df = raw_df.groupby(['year', 'zip_code']).agg(
    crimes_count=pd.NamedAgg(column='DR_NO', aggfunc='count'),
    victims_count=pd.NamedAgg(column='Vict Age', aggfunc=lambda x: sum(x > 0)),
    victims_count_male=pd.NamedAgg(column='Vict Sex', aggfunc=lambda x: sum(x == 'M')),
    victims_count_female=pd.NamedAgg(column='Vict Sex', aggfunc=lambda x: sum(x == 'F')),
    victims_count_age_below_24=pd.NamedAgg(column='Vict Age', aggfunc=lambda x: sum(x <= 24)),
    victims_count_age_between_25_44=pd.NamedAgg(column='Vict Age', aggfunc=lambda x: sum((25 <= x) & (x <= 44))),
    victims_count_age_above_45=pd.NamedAgg(column='Vict Age', aggfunc=lambda x: sum(45 <= x))
).reset_index()
transformed_ethnicity_df = raw_df.groupby(['year', 'zip_code'])['Vict Descent'].value_counts().unstack()
transformed_ethnicity_df = transformed_ethnicity_df.rename(columns=GROUPING_DICT)
transformed_ethnicity_df = transformed_ethnicity_df.T.groupby(level=0).sum().T.reset_index()
transformed_ethnicity_df = transformed_ethnicity_df[['year', 'zip_code'] + list(dict.fromkeys(GROUPING_DICT.values()))]
transformed_df = transformed_df.merge(transformed_ethnicity_df, on=['year', 'zip_code'], how='left')

print('== Storage ==')
transformed_path = os.path.join(TRANSFORMED_DATA_FOLDER_PATH, f'{DATASET_NAME}.csv')
transformed_df.to_csv(transformed_path, index=False)
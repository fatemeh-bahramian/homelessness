import os
import random
import pandas as pd

DATASET_NAME = 'crime'

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
RAW_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', DATASET_NAME)

SAMPLE_SIZE = 10000
TEST_DATA_ROOT = os.path.join(DATA_ROOT, 'test', DATASET_NAME)

raw_file_name_list = list[str]()
for raw_file_name in os.listdir(RAW_DATA_FOLDER_PATH):
    if raw_file_name.endswith('.csv'):
        raw_file_name_list.append(raw_file_name)
raw_file_name_list.sort()

if not os.path.exists(TEST_DATA_ROOT):
    os.makedirs(TEST_DATA_ROOT)

for raw_file_name in raw_file_name_list:

    print(f'== {raw_file_name} ==')

    print('  Loading ...')
    raw_path = os.path.join(RAW_DATA_FOLDER_PATH, raw_file_name)
    raw_df = pd.read_csv(raw_path, low_memory=False)

    print('  Sampling ...')
    test_idx_list = random.sample(range(raw_df.shape[0]), SAMPLE_SIZE)
    test_df = raw_df.iloc[test_idx_list, :]

    print('  Storage ...')
    test_path = os.path.join(TEST_DATA_ROOT, raw_file_name)
    test_df.to_csv(test_path, index=False)
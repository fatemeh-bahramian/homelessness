import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

print('== Summary of Filtered Dataset ==')
filtered_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '02_filtered.csv')
filtered_df = pd.read_csv(filtered_path, low_memory=False)
print('Shape:', filtered_df.shape)
print('Total unique years:', filtered_df['year'].nunique())
print('Total unique ZIP codes:', filtered_df['zip_code'].nunique())

# income_column_list = [column for column in filtered_df.columns if column.startswith('median_income_')]
# filtered_df = filtered_df.drop('year', axis=1)
# for zip_code in list[int](filtered_df['zip_code'].unique()):
#     zip_code_mask = filtered_df['zip_code'] == zip_code
#     for column in income_column_list:
#         income_mask = filtered_df.loc[zip_code_mask, column] < filtered_df.loc[zip_code_mask, column].mean()
#         filtered_df.loc[zip_code_mask & income_mask, column] *= 1000.0

cv_df = filtered_df.groupby(['zip_code']).agg(lambda column_sr: np.nanstd(column_sr) / np.nanmean(column_sr)).reset_index()
melted_df = cv_df.melt(id_vars='zip_code', var_name='column', value_name='cv')

plt.figure(figsize=(16, 9))
scatter = plt.scatter(
    x=melted_df['zip_code'].astype(str), 
    y=melted_df['column'],
    s=3,
    c=melted_df['cv'], 
    cmap='cool',
    alpha=0.8,
    zorder=2
)
plt.colorbar(scatter, label='Coefficient of Variation')
plt.xlabel('ZIP Code', fontsize=8)
plt.ylabel('Column', fontsize=8)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, linestyle='-', alpha=0.7, zorder=0)
plt.tight_layout()
plt.show()
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
PREPARED_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'prepared')

print('== Summary of Filtered Dataset ==')
cleaned_path = os.path.join(PREPARED_DATA_FOLDER_PATH, '03_cleaned.csv')
cleaned_df = pd.read_csv(cleaned_path, low_memory=False)
print('Shape:', cleaned_df.shape)
print('Total unique years:', cleaned_df['year'].nunique())
print('Total unique ZIP codes:', cleaned_df['zip_code'].nunique())

cv_df = cleaned_df.groupby(['zip_code']).agg(lambda column_sr: np.nanstd(column_sr) / np.nanmean(column_sr)).reset_index()
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
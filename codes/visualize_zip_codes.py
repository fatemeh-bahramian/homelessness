import os
import geopandas as gpd
import matplotlib.pyplot as plt

BASE_ROOT = os.getcwd()
DATA_ROOT = os.path.join(BASE_ROOT, 'data')
ZIP_DATA_FOLDER_PATH = os.path.join(DATA_ROOT, 'raw', 'zip')

# Load the ZIP code shapefile (adjust the path accordingly)
zip_path = os.path.join(ZIP_DATA_FOLDER_PATH, 'City_of_Los_Angeles_Zip_Codes.shp')
zip_gdf:gpd.GeoDataFrame = gpd.read_file(zip_path)

# Make sure the shapefile has the correct coordinate reference system (CRS)
# Convert to the appropriate CRS if needed (e.g., WGS84 - EPSG:4326)
zip_gdf = zip_gdf.to_crs(epsg=4326)

print(zip_gdf.shape)
s
# Plot the shapefile
fig, ax = plt.subplots(figsize=(10, 10))
zip_gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5)

# Add the ZIP codes as labels
for _, row in zip_gdf.iterrows():
    ax.text(row['geometry'].centroid.x, row['geometry'].centroid.y,
            str(row['ZCTA5CE10']), fontsize=8, ha='center', color='black')

# Customize the plot (optional)
ax.set_title('Los Angeles ZIP Code Districts', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.axis('equal')  # Equal scaling for both axes

# Show the plot
plt.show()

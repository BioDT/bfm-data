"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# import xarray as xr

# #WORLD BANK DATA INDICATORS
# def plot_agri_2000_smooth(csv_file, output_file):
#     """
#     Plot smooth data for Agri_2000 conforming to country borders.

#     Args:
#         csv_file (str): Path to the input CSV file.
#         shapefile (str): Path to the country shapefile.
#         output_file (str): Path to save the output plot.
#     """
#     # Load CSV data
#     df = pd.read_csv(csv_file)

#     # Ensure longitude is in -180 to 180
#     df["Longitude"] = df["Longitude"].apply(lambda x: x - 360 if x > 180 else x)

#     # Create GeoDataFrame from the CSV
#     gdf = gpd.GeoDataFrame(
#         df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"])
#     )

#     shapefile = natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
#     countries = gpd.read_file(shapefile)

#     # Spatial join to aggregate Agri_2000 data by country borders
#     gdf = gdf[["geometry", "NDVI_08/2010"]]
#     merged = gpd.sjoin(countries, gdf, how="left", predicate="contains")  # Changed `op` to `predicate`

#     # Group by country and calculate the mean Agri_2000 for each country
#     country_agri = merged.groupby("ADMIN")["NDVI_08/2010"].mean().reset_index()

#     # Merge back with the countries GeoDataFrame
#     countries = countries.merge(country_agri, left_on="ADMIN", right_on="ADMIN", how="left")

#     # Plot the data
#     fig, ax = plt.subplots(
#         1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
#     )
#     countries.boundary.plot(ax=ax, linewidth=0.8, edgecolor="black")
#     countries.plot(column="NDVI_08/2010", ax=ax, cmap="viridis", legend=True)

#     # Set extent for Europe and add features
#     ax.set_extent([-25, 45, 34, 72], crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
#     ax.add_feature(cfeature.COASTLINE, edgecolor="black")
#     ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgrey")
#     ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
#     ax.set_title("NDVI (Month August Year 2010) - Europe", fontsize=14)

#     # Save the plot
#     plt.savefig(output_file, dpi=300, bbox_inches="tight")
#     print(f"Plot saved to {output_file}")
#     plt.show()


## ERA5 DATA
# Path to your NetCDF file
# file_path = '/data/projects/biodt/storage/data/Copernicus/ERA5/ERA5-Reanalysis-pressure- 2010-04-15 - 2010-04-15 .nc'

# # Open the NetCDF file using xarray
# ds = xr.open_dataset(file_path)

# # Print the available variables in the dataset to understand the structure
# print(ds)

# # Extract the variable 'z' (height) based on your choice
# variable = ds['q']  # Use 'z' for height

# # Extract latitude, longitude, and time data
# lat = ds['latitude']
# lon = ds['longitude']
# time = ds['valid_time']

# # Select the data for the first time step (adjust the index if needed)
# time_index = 0  # Change this if you want a different time step
# data = variable.isel(valid_time=time_index)

# # Select a specific pressure level (e.g., level 0 for surface)
# pressure_level_index = 0  # Change this based on the pressure level you want to plot
# data_at_level = data.isel(pressure_level=pressure_level_index)

# # Transform longitudes from 0-360 to -180 to 180
# lon_values = lon.values
# lon_transformed = np.where(lon_values > 180, lon_values - 360, lon_values)

# # Update longitude in the dataset to reflect the transformed range
# ds['longitude'] = (('longitude',), lon_transformed)

# # Create a plot of the data for Europe (using transformed longitudes)
# fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# ax.set_extent([-30, 40, 30, 75])  # European bounding box [lon_min, lon_max, lat_min, lat_max]

# # Plot the height data for Europe (using 'z' variable)
# data_at_level.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', levels=20)

# # Add coastlines and gridlines for reference
# ax.coastlines()
# ax.gridlines(draw_labels=True)


# # Title
# plt.title(f'Specific Humidity Data for Europe - {str(time.values[time_index])}')

# # Save the plot as a file (e.g., PNG)
# output_path = 'ERA5_Humidity_Europe_2010-06-26.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()

# # Close the dataset
# ds.close()

# # Print path of saved file
# print(f'Plot saved to: {output_path}')


## Species Distribution


# file_path = "/data/projects/biodt/storage/data/Distribution/species_distribution.csv"
# output_file = "species_distribution_europe_no_legend.png"

# # Load the CSV file
# df = pd.read_csv(file_path)

# # Filter for Europe (longitude between -30 and 60, latitude between 35 and 70)
# df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)
# df_europe = df[(df["lon"] >= -30) & (df["lon"] <= 60) & (df["lat"] >= 35) & (df["lat"] <= 70)]

# # Filter for the year 2010
# df_europe["2010"] = pd.to_numeric(df_europe["2010"], errors="coerce")  # Ensure numeric data
# df_2010 = df_europe[df_europe["2010"].notnull()]

# # Create GeoDataFrame
# gdf = gpd.GeoDataFrame(
#     df_2010, geometry=gpd.points_from_xy(df_2010["lon"], df_2010["lat"])
# )

# # Load the shapefile for country borders
# shapefile = natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
# countries = gpd.read_file(shapefile)

# # Plot the map focused on Europe
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# countries[countries["CONTINENT"] == "Europe"].plot(ax=ax, color="lightgrey", edgecolor="black")

# # Plot detailed points for the species without the legend
# gdf.plot(ax=ax, column="species", cmap="viridis", markersize=20, legend=False)

# # Set limits to focus on Europe
# ax.set_xlim(-30, 60)
# ax.set_ylim(35, 70)

# plt.title("Species Distribution in Europe (2010)", fontsize=16)
# plt.xlabel("Longitude", fontsize=12)
# plt.ylabel("Latitude", fontsize=12)
# plt.savefig(output_file, dpi=300, bbox_inches="tight")
# plt.show()


# #NDVI


# import pandas as pd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import numpy as np

# # Path to your CSV file
# csv_file_path = "/data/projects/biodt/storage/data/Land/Europe_ndvi_test.csv"


# # Load the CSV data
# df = pd.read_csv(csv_file_path)

# # Extract the relevant data
# column = "NDVI_02/2000"  # Column for February 2010
# df = df[["Country", "Latitude", "Longitude", column]]  # Keep only relevant columns
# df = df.dropna(subset=[column])  # Drop rows with NaN NDVI values

# # Convert NDVI values to float (if they are not already)
# df[column] = df[column].astype(float)

# # Transform longitude from 0–360 to -180–180
# df["Longitude"] = np.where(df["Longitude"] > 180, df["Longitude"] - 360, df["Longitude"])

# # Extract latitude, longitude, and NDVI values
# lat = df["Latitude"].values
# lon = df["Longitude"].values
# ndvi = df[column].values

# # Create a scatter plot of NDVI on a map
# fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.set_extent([-30, 40, 30, 75], crs=ccrs.PlateCarree())  # Europe bounding box

# # Add scatter plot for NDVI values
# sc = ax.scatter(
#     lon, lat, c=ndvi, cmap="YlGn", s=10, transform=ccrs.PlateCarree(), vmin=0, vmax=1
# )

# # Add colorbar
# cbar = plt.colorbar(sc, ax=ax, orientation="vertical", label="NDVI")

# # Add features
# ax.coastlines()
# ax.gridlines(draw_labels=True)

# # Add title
# plt.title("NDVI - February 2000")

# # Save the plot (optional)
# plt.savefig("NDVI_February_2000_Map.png", dpi=300, bbox_inches="tight")

# # Show the plot
# plt.show()

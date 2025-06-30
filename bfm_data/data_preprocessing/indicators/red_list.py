import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from bfm_data.config import paths

world_gdf = gpd.read_file("geoBoundaries CGAZ ADM0.geojson").set_crs("EPSG:4326")

df = pd.read_csv(paths.RED_LIST_FILE)
df = df.rename(columns={
    "Entity": "Country",
    "Year": "Year",
    "15.5.1 - Red List Index - ER_RSK_LST": "RedListIndex"
})
df = df.dropna(subset=["Country", "Year", "RedListIndex"])
df["Year"] = df["Year"].astype(int)

european_country_list = [
    'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
    'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Czechia', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
    'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova',
    'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland',
    'Portugal', 'Romania', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain',
    'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican'
]

# =========================
# Choose Region: Europe or Global
# =========================

mode = "europe"

if mode == "europe":
    df = df[df["Country"].isin(european_country_list)]
    gdf = world_gdf[world_gdf["shapeName"].isin(european_country_list)]
else:
    gdf = world_gdf.copy()


available_countries = [c for c in df["Country"].unique() if c in gdf["shapeName"].values]
unmatched = [c for c in df["Country"].unique() if c not in gdf["shapeName"].values]

# =========================
# Generate 0.25° Grid by Country Polygon
# =========================

rows = []
years = list(range(1993, 2025))

for country in available_countries:
    print(f"Processing {country}...")
    
    matching = gdf[gdf["shapeName"] == country]
    if matching.empty:
        continue

    poly = matching.geometry.values[0]
    country_df = df[df["Country"] == country]
    value_by_year = country_df.set_index("Year")["RedListIndex"].to_dict()

    minx, miny, maxx, maxy = poly.bounds
    lats = np.arange(np.floor(miny), np.ceil(maxy), 0.25)
    lons = np.arange(np.floor(minx), np.ceil(maxx), 0.25)

    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            if poly.contains(point):
                row = {
                    "Country": country,
                    "Latitude": round(lat, 10),
                    "Longitude": round(lon, 10)
                }
                for year in years:
                    row[f"RLI_{year}"] = round(value_by_year.get(year, np.nan), 4)
                rows.append(row)


output_df = pd.DataFrame(rows)
filename = "Europe_red_list_index.csv" if mode == "europe" else "Global_red_list_index.csv"
full_output_path = paths.REDLIST_DIR / filename
output_df.to_csv(full_output_path, index=False)
print(f"Saved to '{full_output_path}'")
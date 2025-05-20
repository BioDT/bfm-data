import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from bfm_data.config import paths

mode = "europe"  # or "global"

# Choose one of the indicators below:
# Every time you change Indicator, you must change the filepath to correct csv.
INDICATOR = "Forest area (% of land area)"
FILE_PATH = paths.FOREST_LAND_FILE

prefix_map = {
    "Land area (sq. km)": "Land",
    "Arable land (% of land area)": "Agri",
    "Forest area (% of land area)": "Forest",
    "Agricultural land (% of land area)": "Agri",
    "Agricultural irrigated land (% of total agricultural land)": "Agri",
    "Permanent cropland (% of land area)": "Agri"
}
prefix = prefix_map.get(INDICATOR, "Value")

raw_df = pd.read_csv(FILE_PATH, skiprows=3, nrows=1)
year_cols = [col for col in raw_df.columns if col.strip().isdigit()]

NAME_REPLACEMENTS = {
    "Russian Federation": "Russia",
    "Bahamas, The": "Bahamas",
    "Czech Republic": "Czechia",
    "Korea, Rep.": "South Korea",
    "Korea, Dem. People's Rep.": "North Korea",
    "Gambia, The": "Gambia",
    "Iran, Islamic Rep.": "Iran",
    "Venezuela, RB": "Venezuela",
    "Egypt, Arab Rep.": "Egypt",
    "Syrian Arab Republic": "Syria",
    "Slovak Republic": "Slovakia",
    "Lao PDR": "Laos",
    "Yemen, Rep.": "Yemen",
    "Hong Kong SAR, China": "Hong Kong",
    "Brunei Darussalam": "Brunei",
    "Congo, Dem. Rep.": "Democratic Republic of the Congo",
    "Congo, Rep.": "Republic of the Congo",
    "Macedonia, FYR": "North Macedonia",
    "West Bank and Gaza": "Palestine",
    "Myanmar": "Burma"
}

european_country_list = [
    'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
    'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Czechia', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
    'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova',
    'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland',
    'Portugal', 'Romania', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain',
    'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican'
]

world_gdf = gpd.read_file("geoBoundaries CGAZ ADM0.geojson").set_crs("EPSG:4326")
world_gdf = world_gdf.rename(columns={"shapeName": "GeoCountry"})
world_gdf["GeoCountry_clean"] = world_gdf["GeoCountry"].str.lower().str.strip()

raw_df = pd.read_csv(FILE_PATH, skiprows=3)
df = raw_df[raw_df["Indicator Name"] == INDICATOR]
df = df[["Country Name"] + year_cols]
df = df.rename(columns={"Country Name": "Country"})
df["Country"] = df["Country"].replace(NAME_REPLACEMENTS)
df["Country_clean"] = df["Country"].str.lower().str.strip()

df = df.rename(columns={year: f"{prefix}_{year}" for year in year_cols})

if mode == "europe":
    europe_gdf = world_gdf.cx[-25:45, 34:72]
    valid_countries = [
        c for c in european_country_list
        if c.lower().strip() in df["Country_clean"].values and c.lower().strip() in world_gdf["GeoCountry_clean"].values
    ]
    df = df[df["Country_clean"].isin([c.lower().strip() for c in valid_countries])]
    gdf_subset = world_gdf[world_gdf["GeoCountry_clean"].isin([c.lower().strip() for c in valid_countries])]
else:
    common_names = set(df["Country_clean"]).intersection(world_gdf["GeoCountry_clean"])
    df = df[df["Country_clean"].isin(common_names)]
    gdf_subset = world_gdf[world_gdf["GeoCountry_clean"].isin(common_names)]

rows = []

for name in sorted(df["Country_clean"].unique()):
    geo_row = gdf_subset[gdf_subset["GeoCountry_clean"] == name]
    if geo_row.empty:
        continue

    poly = geo_row.geometry.values[0]
    geo_country = geo_row["GeoCountry"].values[0]
    record = df[df["Country_clean"] == name].iloc[0]
    values = record.drop(["Country", "Country_clean"])

    minx, miny, maxx, maxy = poly.bounds
    lons = np.arange(np.floor(minx), np.ceil(maxx), 0.25)
    lats = np.arange(np.floor(miny), np.ceil(maxy), 0.25)

    print(f"Processing {geo_country}...")

    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            if poly.contains(point):
                rows.append({
                    "Country": geo_country,
                    "Latitude": round(lat, 4),
                    "Longitude": round(lon, 4),
                    **values.to_dict()
                })


filename_base = prefix.lower()
output_filename = f"{mode.capitalize()}_{filename_base}_data.csv"
output_path = paths.FOREST_DIR / output_filename

pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

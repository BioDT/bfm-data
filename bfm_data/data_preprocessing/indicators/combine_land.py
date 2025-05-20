import pandas as pd
from bfm_data.config import paths

mode = "europe"  # or "global"

land_file = f"{mode.capitalize()}_land_data.csv"
ndvi_file = f"{mode.capitalize()}_ndvi_monthly_un_025.csv"

land_df = pd.read_csv(land_file)
ndvi_df = pd.read_csv(ndvi_file)

for df in [land_df, ndvi_df]:
    df["Latitude"] = df["Latitude"].apply(lambda x: round(float(x), 4))
    df["Longitude"] = df["Longitude"].apply(lambda x: round(float(x), 4))

merged_df = pd.merge(
    land_df,
    ndvi_df,
    on=["Country", "Latitude", "Longitude"],
    how="outer",  # Use 'inner' if only common coordinates are needed
    suffixes=("", "_ndvi")
)

output_filename = f"{mode.capitalize()}_combined_land_data.csv"
output_path = paths.LAND_DIR / output_filename
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved to '{output_path}'")
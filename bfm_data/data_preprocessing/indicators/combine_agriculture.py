import pandas as pd
from bfm_data.config import paths

mode = "europe"  # or "global"

input_files = {
    "Agriculture": f"{mode.capitalize()}_agriculture_data.csv",
    "Agriculture_Irrigated": f"{mode.capitalize()}_agriculture_irrigated_data.csv",
    "Arable": f"{mode.capitalize()}_arable_data.csv",
    "Cropland": f"{mode.capitalize()}_cropland_data.csv"
}

all_dataframes = []

for variable_name, filepath in input_files.items():
    df = pd.read_csv(filepath)
    df.insert(0, "Variable", variable_name)
    all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)

output_filename = f"{mode.capitalize()}_combined_agriculture_data.csv"
output_path = paths.AGRICULTURE_DIR / output_filename
combined_df.to_csv(output_path, index=False)

print(f"Saved to '{output_path}'")

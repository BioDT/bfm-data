import torch
 
# Load the .pt file
file_path = "/projects/prjs1134/data/projects/biodt/storage/batches/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt"
data = torch.load(file_path)

# Check the keys in the loaded dictionary
print("Keys in the loaded batch:", data.keys())

# Access individual components
surface_variables = data.get("surface_variables")
single_variables = data.get("single_variables")
atmospheric_variables = data.get("atmospheric_variables")
species_variables = data.get("species_variables")
species_extinction_variables = data.get("species_extinction_variables")
land_variables = data.get("land_variables")
agriculture_variables = data.get("agriculture_variables")
forest_variables = data.get("forest_variables")
batch_metadata = data.get("batch_metadata")

# Display metadata
if batch_metadata:
    print("Metadata latitudes:", batch_metadata.get("latitudes"))
    print("Metadata longitudes:", batch_metadata.get("longitudes"))
    print("Metadata timestamp:", batch_metadata.get("timestamp"))
    print("Metadata pressure levels:", batch_metadata.get("pressure_levels"))
    print("Metadata species list:", batch_metadata.get("species_list"))

# Optional: Check the contents of a specific variable
print("Surface Variables:", surface_variables)
print({k: v.shape for k,v in surface_variables.items()})
print({k: v.shape for k,v in single_variables.items()})
print({k: v.shape for k,v in atmospheric_variables.items()})
print({k: v.shape for k,v in species_variables.items()})
print({k: v.shape for k,v in species_extinction_variables.items()})
print({k: v.shape for k,v in land_variables.items()})
print({k: v.shape for k,v in land_variables.items()})
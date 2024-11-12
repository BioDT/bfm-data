# src/utils/statistics.py

import csv
import logging
import os
import warnings
from collections import Counter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from geopy.geocoders import Nominatim
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

from src.config import paths
from src.dataset_creation.load_data import load_species_data

warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")


def reverse_geocode(lat: float, lon: float) -> tuple:
    """
    Perform reverse geocoding to convert latitude and longitude coordinates into
    a country and city name.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        tuple: A tuple containing the country and city as strings.
               If the location cannot be resolved, returns ("Unknown", "Unknown").
    """
    geolocator = Nominatim(user_agent="geoapi")

    try:
        location = geolocator.reverse((lat, lon), language="en", timeout=10)

        if location:
            address = location.raw.get("address", {})
            country = address.get("country", "Unknown")
            city = address.get("city", "Unknown")
            return country, city

        return "Unknown", "Unknown"

    except Exception as e:
        logging.error(f"Error during reverse geocoding: {e}")
        return "Unknown", "Unknown"


def dataset_statistics(species_file: str) -> None:
    """
    Provide statistics of the dataset including the total number of data points (individuals)
    and the number of entries per modality.

    Args:
        species_file (str): Path to the Parquet file containing species data.
    """
    species_dataset = load_species_data(species_file)

    return {
        "Total data points": len(species_dataset),
        "Entries with Images": species_dataset["Image"].notna().sum(),
        "Entries with Audio": species_dataset["Audio"].notna().sum(),
        "Entries with eDNA": species_dataset["eDNA"].notna().sum(),
        "Entries with Descriptions": species_dataset["Description"].notna().sum(),
    }


def count_points_by_country_city(
    species_file: str, start_date: str = None, end_date: str = None
) -> tuple[dict, dict]:
    """
    Count how many data points are in each country and city based on latitude and longitude,
    filtered by the given start and end dates.

    Args:
        species_file (str): Path to the Parquet file containing species data.
        start_date (str): Start date for filtering in the format 'YYYY-MM-DD'.
        end_date (str): End date for filtering in the format 'YYYY-MM-DD'.

    Returns:
        None.
    """
    species_dataset = load_species_data(species_file)

    species_dataset["Timestamp"] = species_dataset["Timestamp"].apply(
        lambda x: x[0] if isinstance(x, tuple) else x
    )
    species_dataset["Timestamp"] = pd.to_datetime(species_dataset["Timestamp"])

    if start_date:
        start_date = pd.to_datetime(start_date, dayfirst=True)
        species_dataset = species_dataset[species_dataset["Timestamp"] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date, dayfirst=True)
        species_dataset = species_dataset[species_dataset["Timestamp"] <= end_date]

    species_dataset = species_dataset.dropna(subset=["Latitude", "Longitude"])

    country_counter = Counter()
    city_counter = Counter()

    for _, row in species_dataset.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        country, city = reverse_geocode(lat, lon)
        country_counter[country] += 1
        city_counter[city] += 1

    return country_counter, city_counter


def get_species_distribution_by_class(species_file: str) -> None:
    """
    Calculate the distribution of species by class in the dataset.

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        None.
    """
    species_dataset = load_species_data(species_file)

    if "Genus" in species_dataset.columns and not species_dataset["Genus"].isna().all():
        print(species_dataset["Genus"])
        return species_dataset["Genus"].value_counts()
    else:
        print("The dataset does not contain any data in the 'Class' column.")
        return pd.Series(dtype=int)


def get_unique_filename(output_dir: str, base_name: str) -> str:
    """
    Generate a unique filename by appending a counter if the file already exists.

    Args:
        output_dir (str): The directory to save the file.
        base_name (str): The base name of the file without extension.

    Returns:
        str: A unique file path with a counter appended if necessary.
    """
    counter = 1
    file_name = f"{base_name}.png"
    file_path = os.path.join(output_dir, file_name)

    while os.path.exists(file_path):
        file_name = f"{base_name}_{counter}.png"
        file_path = os.path.join(output_dir, file_name)
        counter += 1

    return file_path


def visualize_data_on_map(
    species_file: str, start_date: str = None, end_date: str = None
) -> None:
    """
    Visualize the selected data modalities (e.g., images, audio, etc.) on a map,
    filtered by timestamp, and save the map in the STATISTICS folder with a unique name.

    Args:
        species_file (str): Path to the Parquet file containing species data.
        start_date (str): Start date for filtering in the format 'YYYY-MM-DD'.
        end_date (str): End date for filtering in the format 'YYYY-MM-DD'.
    """
    species_dataset = load_species_data(species_file)

    species_dataset["Timestamp"] = species_dataset["Timestamp"].apply(
        lambda x: x[0] if isinstance(x, tuple) else x
    )
    species_dataset["Timestamp"] = pd.to_datetime(species_dataset["Timestamp"])

    if start_date:
        start_date = pd.to_datetime(start_date)
        species_dataset = species_dataset[species_dataset["Timestamp"] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        species_dataset = species_dataset[species_dataset["Timestamp"] <= end_date]

    species_dataset = species_dataset.dropna(subset=["Latitude", "Longitude"])

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    min_lon, max_lon = (
        species_dataset["Longitude"].min(),
        species_dataset["Longitude"].max(),
    )
    min_lat, max_lat = (
        species_dataset["Latitude"].min(),
        species_dataset["Latitude"].max(),
    )
    ax.set_extent(
        [min_lon - 2, max_lon + 2, min_lat - 2, max_lat + 2], crs=ccrs.PlateCarree()
    )

    land = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="black", facecolor=cfeature.COLORS["land"]
    )
    borders = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        "10m",
        edgecolor="black",
        facecolor="none",
    )

    ax.add_feature(land, zorder=0)
    ax.add_feature(borders, linewidth=1.5, zorder=1)

    ax.scatter(
        species_dataset["Longitude"],
        species_dataset["Latitude"],
        color="red",
        label="Species Data",
        s=10,
        zorder=2,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(
        f"Species Data from {start_date.date()} to {end_date.date()}", fontsize=14
    )

    output_dir = paths.STATISTICS_DIR
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"species_map_{start_date.date()}_{end_date.date()}"

    output_file = get_unique_filename(output_dir, base_name)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Map saved as {output_file}")


def calculate_species_richness(species_file: str) -> int:
    """
    Calculate the species richness, i.e., the number of unique species in the dataset.

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        unique_species (int): The number of unique species.
    """
    species_dataset = load_species_data(species_file)
    unique_species = species_dataset["Species"].nunique()
    return unique_species


def calculate_shannon_index(species_file: str) -> float:
    """
    Calculate Shannon's Diversity Index (H') for the dataset, which measures species diversity
    by accounting for both species richness and evenness.

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        float: Shannon's diversity index, a measure of diversity in the dataset.
    """
    species_dataset = load_species_data(species_file)
    species_counts = species_dataset["Species"].value_counts()

    proportions = species_counts / species_counts.sum()

    return entropy(proportions)


def calculate_simpson_index(species_file: str) -> float:
    """
    Calculate Simpson's Diversity Index (D), which measures the probability that two
    individuals randomly selected from a sample will belong to the same species.

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        float: Simpson's diversity index, where lower values indicate higher biodiversity.
    """
    species_dataset = load_species_data(species_file)
    species_counts = species_dataset["Species"].value_counts()

    proportions = species_counts / species_counts.sum()
    simpson_index = 1 - sum(proportions**2)

    return simpson_index


def add_region_info(
    species_file: str, lat_col: str = "Latitude", lon_col: str = "Longitude"
) -> pd.DataFrame:
    """
    Add region information (e.g., Country) to the species dataset using reverse geocoding.

    Args:
        species_file (str): Path to the Parquet file containing species data.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.

    Returns:
        pd.DataFrame: The species dataset with an additional 'Region' column.
    """
    species_dataset = load_species_data(species_file)
    species_dataset["Region"] = species_dataset.apply(
        lambda row: reverse_geocode(row[lat_col], row[lon_col]), axis=1
    )

    return species_dataset


def calculate_endemism(species_file: str) -> pd.DataFrame:
    """
    Calculate endemism, identifying species unique to specific regions (e.g., countries).

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        pd.DataFrame: A summary of endemism showing how many species are unique to each region.
    """
    species_dataset = load_species_data(species_file)
    if "Region" not in species_dataset.columns:
        species_dataset = add_region_info(species_dataset)

    grouped = (
        species_dataset.groupby(["Species", "Region"]).size().reset_index(name="Count")
    )

    endemism = grouped.groupby("Species").filter(lambda x: len(x) == 1)

    endemism_summary = (
        endemism.groupby("Region").size().reset_index(name="Endemic_Species_Count")
    )

    return endemism_summary


def calculate_beta_diversity(species_file: str) -> pd.DataFrame:
    """
    Calculate beta diversity (species turnover) between different regions.

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        pd.DataFrame: A summary of beta diversity between regions.
    """

    species_dataset = load_species_data(species_file)

    if "Region" not in species_dataset.columns:
        species_dataset = add_region_info(species_dataset)

    species_matrix = pd.pivot_table(
        species_dataset, index="Region", columns="Species", aggfunc="size", fill_value=0
    )

    beta_div = pdist(species_matrix.values, metric="jaccard")
    beta_div_matrix = squareform(beta_div)

    beta_div_df = pd.DataFrame(
        beta_div_matrix, index=species_matrix.index, columns=species_matrix.index
    )

    return beta_div_df


def analyze_climate_biodiversity(
    species_file: str, climate_data: xr.Dataset, indicator
) -> pd.DataFrame:
    """
    Analyze biodiversity in relation to a specific climate indicator.

    Args:
        species_file (str): Path to the Parquet file containing species data.
        climate_data (xr.Dataset): xarray Dataset containing ERA5 climate data.
        indicator (str): Climate indicator to analyze (e.g., 'temperature', 'precipitation').

    Returns:
        pd.DataFrame: A DataFrame summarizing biodiversity and the selected climate indicator.
    """
    species_dataset = load_species_data(species_file)
    species_dataset["Country"], species_dataset["City"] = zip(
        *species_dataset.apply(
            lambda row: reverse_geocode(row["Latitude"], row["Longitude"]), axis=1
        )
    )

    biodiversity_stats = (
        species_dataset.groupby("Country")["Species"]
        .nunique()
        .reset_index(name="Species_Richness")
    )

    climate_values = climate_data[indicator].mean(dim=["latitude", "longitude"]).values
    biodiversity_stats[indicator] = climate_values[: len(biodiversity_stats)]

    return biodiversity_stats


def analyze_biodiversity(species_file: str) -> None:
    """
    Perform biodiversity analysis on the species dataset by calculating species richness,
    Shannon's index, Simpson's index, and functional diversity.

    Args:
        species_dataset (pd.DataFrame): The dataset containing species and environmental data.
    """

    species_richness = calculate_species_richness(species_file)
    print(f"Species Richness: {species_richness}")

    shannon_index = calculate_shannon_index(species_file)
    print(f"Shannon's Diversity Index: {shannon_index}")

    simpson_index = calculate_simpson_index(species_file)
    print(f"Simpson's Diversity Index: {simpson_index}")


def analyze_directory(base_directory: str) -> None:
    """
    Analyze the directory structure to count the occurrences of images, audio files,
    eDNA files, description files, and distribution files. Additionally, categorize
    folders based on the types of files they contain, and write a summary report
    to 'directory_analysis.txt'.

    Args:
        base_directory (str): The root directory containing species and environmental data.
    """

    total_folders = 0
    image_count = 0
    audio_count = 0
    edna_count = 0
    description_count = 0
    distribution_count = 0
    condition_counts = Counter()

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    audio_extensions = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}

    for root, _, files in os.walk(base_directory):
        if root != base_directory:
            total_folders += 1  # Increment for each folder encountered

        has_images = has_audio = has_edna = False

        for file in files:
            _ = os.path.join(root, file)
            extension = os.path.splitext(file)[1].lower()

            if extension in image_extensions:
                has_images = True
                image_count += 1

            elif extension in audio_extensions:
                has_audio = True
                audio_count += 1

            elif file.endswith("_edna.csv"):
                has_edna = True
                edna_count += 1
            elif file.endswith("_description.csv"):
                description_count += 1
            elif file.endswith("_distribution.csv"):
                distribution_count += 1

        if has_images and not has_audio and not has_edna:
            condition_counts["folders_with_images_no_audio_no_edna"] += 1
        if has_audio and not has_images and not has_edna:
            condition_counts["folders_with_audio_no_images_no_edna"] += 1
        if has_edna and not has_images and not has_audio:
            condition_counts["folders_with_edna_no_images_no_audio"] += 1
        if has_images and has_audio and not has_edna:
            condition_counts["folders_with_images_audio_no_edna"] += 1
        if has_images and has_edna and not has_audio:
            condition_counts["folders_with_images_edna_no_audio"] += 1
        if has_audio and has_edna and not has_images:
            condition_counts["folders_with_audio_edna_no_images"] += 1
        if has_images and has_audio and has_edna:
            condition_counts["folders_with_all_files"] += 1

    with open(
        f"{paths.STATISTICS_DIR}/directory_analysis.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["Category", "Count"])

        writer.writerow(["Total folders", total_folders])
        writer.writerow(["Total images", image_count])
        writer.writerow(["Total audios", audio_count])
        writer.writerow(["Total '_edna.csv' files", edna_count])
        writer.writerow(["Total '_description.csv' files", description_count])
        writer.writerow(["Total '_distribution.csv' files", distribution_count])

        for condition, count in condition_counts.items():
            writer.writerow([condition.replace("_", " ").capitalize(), count])

    print("Analysis completed and saved to 'directory_analysis.csv'.")


def analyze_dataset(
    species_file: str, start_date: str = None, end_date: str = None
) -> None:
    """
    Generate a CSV report for dataset statistics, including modality counts, species distribution by class,
    and data points by country and city.

    Args:
        species_file (str): Path to the Parquet file containing species data.
        start_date (str): Start date for filtering in the format 'YYYY-MM-DD'.
        end_date (str): End date for filtering in the format 'YYYY-MM-DD'.
    """
    dataset_stats = dataset_statistics(species_file)
    class_distribution = get_species_distribution_by_class(species_file)
    country_counter, city_counter = count_points_by_country_city(
        species_file, start_date, end_date
    )

    with open(
        f"{paths.STATISTICS_DIR}/dataset_statistics.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Count"])

        for key, value in dataset_stats.items():
            writer.writerow([key, value])

        writer.writerow([])
        writer.writerow(["Species distribution by class"])
        for species_class, count in class_distribution.items():
            writer.writerow([species_class, count])

        writer.writerow([])
        writer.writerow(["Points by Country"])
        for country, count in country_counter.items():
            writer.writerow([country, count])

        writer.writerow([])
        writer.writerow(["Points by City"])
        for city, count in city_counter.items():
            writer.writerow([city, count])

    print("Dataset analysis completed and saved to 'dataset_statistics.csv'.")

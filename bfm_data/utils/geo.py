"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import country_bounding_boxes as cbb
import country_converter as coco
import pycountry


def clean_iso_codes(iso2_codes: list) -> list:
    """
    Cleans ISO country codes by standardizing country code patterns.

    This function addresses specific patterns such as '^GR$|^EL$' and '^GB$|^UK$'
    to ensure consistency in ISO code representation. Other codes remain unchanged.

    Args:
        iso2_codes (list): A list of ISO2 country codes.

    Returns:
        list: A cleaned list of ISO2 country codes with standard representations.
    """
    cleaned_codes = []
    for code in iso2_codes:
        if code in ["^GR$|^EL$", "GR", "EL"]:
            cleaned_codes.append("GR")
        elif code in ["^GB$|^UK$", "GB", "UK"]:
            cleaned_codes.append("GB")
        else:
            cleaned_codes.append(code)
    return cleaned_codes


def get_countries_by_continent(continent_name: str) -> tuple[list, list]:
    """
    Retrieves a list of countries and their ISO codes based on a specified continent.

    This function uses the `country_converter` library to filter countries by continent
    and converts their ISO codes to standard names using the `pycountry` library.

    Args:
        continent_name (str): The name of the continent (e.g., 'Europe', 'South America').

    Returns:
        tuple: A tuple containing:
            - list of country names within the specified continent.
            - list of ISO2 country codes for countries in the specified continent.
    """
    converter = coco.CountryConverter()

    all_iso2_codes = converter.data["ISO2"].tolist()
    cleaned_iso2_codes = clean_iso_codes(all_iso2_codes)

    continent_countries = []
    continent_isos = []
    for iso2_code in cleaned_iso2_codes:
        try:
            continent = converter.convert(iso2_code, to="continent")
            if continent == continent_name:
                country = pycountry.countries.get(alpha_2=iso2_code)
                if country:
                    continent_countries.append(country.name)
                    continent_isos.append(iso2_code)
        except Exception as e:
            print(f"Error processing ISO code {iso2_code}: {e}")

    return continent_countries, continent_isos


def get_bounding_boxes_for_countries(iso_codes: list) -> dict:
    """
    Retrieves bounding boxes for a list of countries using their ISO codes.
    For countries with multiple subregions, aggregates subregion bounding boxes.

    Args:
        iso_codes (list): A list of ISO2 country codes.

    Returns:
        dict: A dictionary where the keys are ISO codes or country names,
              and the values are tuples representing bounding box coordinates
              (min longitude, min latitude, max longitude, max latitude).
    """
    country_bboxes = {}

    for iso in iso_codes:
        min_lon, min_lat = float("inf"), float("inf")
        max_lon, max_lat = float("-inf"), float("-inf")

        found_subregion = False
        for country in cbb.country_subunits_by_iso_code(iso):
            found_subregion = True
            (
                country_min_lon,
                country_min_lat,
                country_max_lon,
                country_max_lat,
            ) = country.bbox
            min_lon = min(min_lon, country_min_lon)
            min_lat = min(min_lat, country_min_lat)
            max_lon = max(max_lon, country_max_lon)
            max_lat = max(max_lat, country_max_lat)

        if found_subregion:
            country_bboxes[iso] = (min_lon, min_lat, max_lon, max_lat)

    return country_bboxes


def get_country_bounding_box(country_or_iso: str) -> tuple:
    """
    Retrieves bounding box for a given country using its ISO code or name.

    Args:
        country_or_iso (str): The country name or ISO code.

    Returns:
        tuple: Bounding box coordinates (min_lon, min_lat, max_lon, max_lat) or None if not found.
    """
    if len(country_or_iso) == 2 and country_or_iso.isalpha():
        iso_code = country_or_iso.upper()
    else:
        iso_code = coco.convert(names=country_or_iso, to="ISO2")

    country_name = pycountry.countries.get(alpha_2=iso_code).name if iso_code else None

    bounding_boxes = get_bounding_boxes_for_countries([iso_code])

    return bounding_boxes.get(iso_code) or bounding_boxes.get(country_name)


def get_country_name_from_iso(iso_code: str) -> str:
    """
    Get the country name from its ISO code.

    Args:
        iso_code (str): The ISO code of the country (e.g., 'FR').

    Returns:
        str: The name of the country, or 'Unknown' if not found.
    """
    try:
        country = pycountry.countries.get(alpha_2=iso_code.upper())
        return country.name if country else "Unknown"
    except Exception as e:
        print(f"Error retrieving country name for ISO code {iso_code}: {e}")
        return "Unknown"

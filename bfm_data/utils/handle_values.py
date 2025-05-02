"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

from datetime import datetime, timedelta, timezone


def parse_date_time(date: str, time: str, timezone_offset: str = "+00:00") -> str:
    """
    Parses a date and time string into a standardized ISO 8601 timestamp.

    This function handles various non-standard time formats, including:
    - AM/PM formats (e.g., "9.20 pm")
    - Descriptive times (e.g., "Dawn", "Midday", "Sunset")
    - Placeholder times (e.g., "xx:xx")
    - Invalid year or day/month placeholders (e.g., "0000-00-00")

    Args:
        date (str): The date string in the format 'YYYY-MM-DD'.
        time (str): The time string, which may include non-standard formats.

    Returns:
        str: ISO 8601 formatted timestamp if parsing is successful.
        None: If the date or time is invalid or cannot be parsed.
    """

    time_mappings = {
        "Midday": "12:00",
        "Dawn": "06:00",
        "Dawn (at dusk)": "06:00",
        "Sunset (at dusk)": "18:00",
        "Sunset": "18:00",
        "Night": "20:00",
        "dusk": "18:00",
        "early": "06:00",
    }

    def normalize_time(time):
        """
        Normalizes the time string by handling AM/PM formats, descriptive times, and placeholders.

        Args:
            time_str (str): The time string to normalize.

        Returns:
            str: A normalized time string in 24-hour format or '00:00' for placeholders.
        """
        if any(marker in time for marker in ["xx:xx", "?:?", "x"]):
            return "00:00"

        if time == "Unknown":
            time = "00:00"

        time = time.replace(".", ":")

        if time.lower() == "am":
            return "00:00"
        if time.lower() == "pm":
            return "12:00"

        if any(marker in time for marker in [" am", " pm"]):
            time = time.replace(" am", "AM").replace(" pm", "PM").strip()
            try:
                time = datetime.strptime(time, "%I:%M%p").strftime("%H:%M")
            except ValueError:
                return "00:00"

        if time in time_mappings:
            return time_mappings[time]

        if len(time) == 2 and time.isdigit():
            return time + ":00"

        try:
            hours, minutes = map(int, time.split(":"))
            if hours >= 24 or minutes >= 60:
                return "00:00"
        except ValueError:
            return "00:00"

        return time

    if date.startswith("0000"):
        return None

    if "-00" in date:
        date = date.replace("-00", "-01")

    time = normalize_time(time)

    try:
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        hours_offset, minutes_offset = map(int, timezone_offset.split(":"))
        tz = timezone(timedelta(hours=hours_offset, minutes=minutes_offset))
        dt = dt.replace(tzinfo=tz)
        return dt.isoformat()

    except ValueError as e:
        print(f"Error parsing date and time: {e}")
        return None

from datetime import datetime

def get_day_of_year(date_str: str = None) -> int:
    """
    Returns the day number of the year for a given date string (YYYY-MM-DD).
    If no date is provided, it uses the current date.

    Args:
        date_str (str, optional): Date in YYYY-MM-DD format. Defaults to today's date.

    Returns:
        int: The day of the year (1-365 or 1-366 in leap years).
    """
    try:
        if date_str:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            dt = datetime.today()  # Use current date if none provided

        return dt.timetuple().tm_yday

    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD.")
        return -1  # Return -1 to indicate an error

import os
from typing import List

def transform_date(date_str: str) -> str:
    """
    Transforms a ROC date string (e.g., '112/01/01') to a Gregorian date string (e.g., '2023/01/01').
    If the year is already in Gregorian format (e.g., 4 digits or > 1911), it returns the original string.
    
    Args:
        date_str (str): Date string in ROC format (YYY/MM/DD) or Gregorian.

    Returns:
        str: Date string in Gregorian format (YYYY/MM/DD), or original string if format is invalid.
    """
    try:
        parts = date_str.split('/')
        if len(parts) != 3:
            return date_str
        
        y_str, m, d = parts
        
        # Check if year is numeric
        if not y_str.isdigit():
            return date_str
            
        y = int(y_str)
        
        # Heuristic: If year is already greater than 1911 or has 4 digits, assume it's Gregorian
        if y > 1911 or len(y_str) == 4:
            return date_str
            
        return f"{y + 1911}/{m}/{d}"
    except ValueError:
        return date_str

def generate_date_list(start_year: int, start_month: int, end_year: int, end_month: int) -> List[str]:
    """
    Generates a list of date strings for the 1st of each month between the start and end dates.
    Format: 'YYYYMM01' (used for TWSE URL queries).

    Args:
        start_year (int): Starting year (e.g., 2023).
        start_month (int): Starting month (1-12).
        end_year (int): Ending year.
        end_month (int): Ending month.

    Returns:
        List[str]: List of date strings, e.g., ['20230101', '20230201', ...].
    """
    dates = []
    
    # Simple validation to avoid infinite loops if end < start
    if start_year > end_year or (start_year == end_year and start_month > end_month):
        return []

    for year in range(start_year, end_year + 1):
        s_m = start_month if year == start_year else 1
        e_m = end_month if year == end_year else 12
        
        for month in range(s_m, e_m + 1):
            dates.append(f"{year}{month:02d}01")
            
    return dates

def ensure_dir_exists(directory: str):
    """
    Ensures that a directory exists. If not, creates it.
    
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
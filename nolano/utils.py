import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Literal


def array_split(arr: List[Any], n: int = 100) -> List[List[Any]]:
    """Split array into chunks.
    
    Args:
        arr (List[Any]): Input list or array-like object.
        n (int): Maximum chunk size. Default 100.
    
    Returns:
        List[List[Any]]: List of chunks.
    """
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def convert_sulie_to_nolano_frequency(sulie_freq: str) -> str:
    """Convert Sulie frequency notation to Nolano frequency.
    
    Args:
        sulie_freq (str): Sulie frequency notation ("H", "D", "W", "M", "Y")
        
    Returns:
        str: Nolano frequency string
        
    Raises:
        ValueError: If frequency mapping not found
    """
    frequency_mapping = {
        "H": "Hours",
        "D": "Daily", 
        "W": "Weekly",
        "M": "Monthly",
        "Y": "Yearly"
    }
    
    if sulie_freq not in frequency_mapping:
        raise ValueError(f"Unsupported frequency: {sulie_freq}. "
                        f"Supported: {list(frequency_mapping.keys())}")
    
    return frequency_mapping[sulie_freq]


def convert_nolano_to_sulie_frequency(nolano_freq: str) -> str:
    """Convert Nolano frequency notation to Sulie frequency.
    
    Args:
        nolano_freq (str): Nolano frequency string 
        
    Returns:
        str: Sulie frequency notation
        
    Raises:
        ValueError: If frequency mapping not found
    """
    frequency_mapping = {
        "Seconds": "S",
        "Minutes": "T", 
        "Hours": "H",
        "Daily": "D",
        "Weekly": "W", 
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }
    
    if nolano_freq not in frequency_mapping:
        raise ValueError(f"Unsupported frequency: {nolano_freq}. "
                        f"Supported: {list(frequency_mapping.keys())}")
    
    return frequency_mapping[nolano_freq]


def forecast_to_nolano_format(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str
) -> Dict[str, Any]:
    """Convert pandas DataFrame to Nolano API series format.
    
    Args:
        df (pd.DataFrame): Input DataFrame with time series data
        timestamp_col (str): Column name containing timestamps
        value_col (str): Column name containing values
        
    Returns:
        Dict[str, Any]: Nolano series format with timestamps and values
        
    Raises:
        KeyError: If specified columns not found
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found")
    
    if value_col not in df.columns:
        raise KeyError(f"Value column '{value_col}' not found")
    
    # Sort by timestamp and convert to required format
    df_sorted = df.sort_values(timestamp_col)
    timestamps = pd.to_datetime(df_sorted[timestamp_col])
    timestamp_strings = timestamps.dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    values = df_sorted[value_col].astype(float).tolist()
    
    return {
        'timestamps': timestamp_strings,
        'values': values
    }


def nolano_forecast_to_dataframe(
    forecast_timestamps: List[str],
    lower_bound: List[float],
    median: List[float], 
    upper_bound: List[float]
) -> pd.DataFrame:
    """Convert Nolano forecast results to pandas DataFrame.
    
    Args:
        forecast_timestamps (List[str]): Forecast timestamps
        lower_bound (List[float]): Lower confidence bounds
        median (List[float]): Median forecast values
        upper_bound (List[float]): Upper confidence bounds
        
    Returns:
        pd.DataFrame: DataFrame with forecast results
        
    Raises:
        ValueError: If arrays have different lengths
    """
    lengths = [len(forecast_timestamps), len(lower_bound), len(median), len(upper_bound)]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All forecast arrays must have the same length")
    
    return pd.DataFrame({
        'timestamp': pd.to_datetime(forecast_timestamps),
        'lower_bound': lower_bound,
        'median': median, 
        'upper_bound': upper_bound
    })


def convert_quantiles_to_confidence(quantiles: List[float]) -> float:
    """Convert Sulie quantile range to Nolano confidence level.
    
    Args:
        quantiles (List[float]): Quantiles (e.g., [0.1, 0.9])
        
    Returns:
        float: Confidence level (e.g., 0.8 for [0.1, 0.9])
        
    Raises:
        ValueError: If quantiles format is invalid
    """
    if len(quantiles) != 2:
        raise ValueError("Exactly two quantiles required (lower and upper)")
    
    lower, upper = sorted(quantiles)
    
    if not (0 <= lower < upper <= 1):
        raise ValueError("Quantiles must be between 0 and 1, with lower < upper")
    
    confidence = upper - lower
    return confidence


def convert_confidence_to_quantiles(confidence: float) -> List[float]:
    """Convert Nolano confidence level to Sulie quantile range.
    
    Args:
        confidence (float): Confidence level (e.g., 0.8)
        
    Returns:
        List[float]: Quantiles (e.g., [0.1, 0.9])
        
    Raises:
        ValueError: If confidence level is invalid
    """
    if not (0 < confidence < 1):
        raise ValueError("Confidence must be between 0 and 1")
    
    tail = (1 - confidence) / 2
    lower = tail
    upper = 1 - tail
    
    return [lower, upper]


def validate_nolano_series_format(series: List[Dict[str, Any]]) -> bool:
    """Validate that data is in proper Nolano series format.
    
    Args:
        series (List[Dict]): List of series objects
        
    Returns:
        bool: True if format is valid
        
    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(series, list) or len(series) == 0:
        raise ValueError("Series must be a non-empty list")
    
    for i, s in enumerate(series):
        if not isinstance(s, dict):
            raise ValueError(f"Series {i} must be a dictionary")
        
        if 'timestamps' not in s or 'values' not in s:
            raise ValueError(f"Series {i} must have 'timestamps' and 'values' keys")
        
        if not isinstance(s['timestamps'], list) or not isinstance(s['values'], list):
            raise ValueError(f"Series {i}: timestamps and values must be lists")
        
        if len(s['timestamps']) != len(s['values']):
            raise ValueError(f"Series {i}: timestamps and values must have same length")
        
        if len(s['timestamps']) == 0:
            raise ValueError(f"Series {i}: must have at least one data point")
    
    return True
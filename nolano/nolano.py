import json
import requests
from typing import List, Optional, Dict, Any, Literal, Union
from dataclasses import dataclass
import pandas as pd
from datetime import datetime


@dataclass
class NolanoForecast:
    """A class for Nolano time series forecasting results."""
    forecast_timestamps: List[str]
    lower_bound: List[float]
    median: List[float]
    upper_bound: List[float]
    
    def __post_init__(self):
        """Validate input data."""
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input data dimensions."""
        lengths = [
            len(self.forecast_timestamps),
            len(self.lower_bound),
            len(self.median),
            len(self.upper_bound)
        ]
        
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All forecast arrays must have the same length")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to pandas DataFrame."""
        return pd.DataFrame({
            'timestamp': pd.to_datetime(self.forecast_timestamps),
            'lower_bound': self.lower_bound,
            'median': self.median,
            'upper_bound': self.upper_bound
        })


class NolanoClient:
    """Client for interacting with Nolano's time series forecasting API."""
    
    BASE_URL = "https://api.nolano.ai"
    
    # Available Nolano models
    AVAILABLE_MODELS = [
        "forecast-model-1",  # Primary Forecasting Model
        "forecast-model-2",  # Alternative Forecasting Model  
        "forecast-model-3",  # Advanced Forecasting Model
        "forecast-model-4"   # Next-Generation Model
    ]
    
    VALID_FREQUENCIES = [
        "Seconds", "Minutes", "Hours", "Daily", 
        "Weekly", "Monthly", "Quarterly", "Yearly"
    ]
    
    def __init__(self, api_key: str, model_id: Optional[str] = None):
        """Initialize Nolano client with API credentials.
        
        Args:
            api_key (str): API key for authentication
            model_id (str, optional): Default model ID to use. Defaults to 'forecast-model-1'
        
        Raises:
            ValueError: If model_id is not in available models
        """
        self.api_key = api_key
        self.model_id = model_id or "forecast-model-1"
        
        if self.model_id not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model ID must be one of: {self.AVAILABLE_MODELS}")
    
    def _get_headers(self, model_id: Optional[str] = None) -> Dict[str, str]:
        """Get request headers for API calls.
        
        Args:
            model_id (str, optional): Override default model ID
            
        Returns:
            Dict[str, str]: Headers for API request
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }
        
        if model_id or self.model_id:
            headers['X-Model-Id'] = model_id or self.model_id
            
        return headers
    
    def forecast(
        self,
        series: List[Dict[str, Any]],
        forecast_horizon: int,
        data_frequency: str,
        forecast_frequency: str,
        confidence: float = 0.95,
        model_id: Optional[str] = None
    ) -> NolanoForecast:
        """Generate a time series forecast using Nolano API.
        
        Args:
            series (List[Dict]): List containing time series data. Each dict should have
                'timestamps' (List[str]) and 'values' (List[float]) keys
            forecast_horizon (int): Number of future periods to predict
            data_frequency (str): Frequency of input data (e.g., "Daily", "Hourly")
            forecast_frequency (str): Desired forecast frequency (must match data_frequency)
            confidence (float): Confidence level between 0 and 1. Defaults to 0.95
            model_id (str, optional): Override default model ID
            
        Returns:
            NolanoForecast: Forecast results with timestamps and prediction intervals
            
        Raises:
            ValueError: If parameters are invalid
            requests.exceptions.HTTPError: If API request fails
        """
        # Validate inputs
        if not series:
            raise ValueError("At least one time series is required")
        
        if len(series) > 1:
            raise ValueError("Nolano API currently supports only one time series")
        
        if data_frequency not in self.VALID_FREQUENCIES:
            raise ValueError(f"data_frequency must be one of: {self.VALID_FREQUENCIES}")
        
        if forecast_frequency not in self.VALID_FREQUENCIES:
            raise ValueError(f"forecast_frequency must be one of: {self.VALID_FREQUENCIES}")
        
        if data_frequency != forecast_frequency:
            raise ValueError("forecast_frequency must match data_frequency")
        
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        
        # Validate series structure
        for i, s in enumerate(series):
            if 'timestamps' not in s or 'values' not in s:
                raise ValueError(f"Series {i} must have 'timestamps' and 'values' keys")
            
            if len(s['timestamps']) != len(s['values']):
                raise ValueError(f"Series {i}: timestamps and values must have same length")
        
        # Prepare request payload
        payload = {
            "series": series,
            "forecast_horizon": forecast_horizon,
            "data_frequency": data_frequency,
            "forecast_frequency": forecast_frequency,
            "confidence": confidence
        }
        
        # Make API request
        headers = self._get_headers(model_id)
        url = f"{self.BASE_URL}/forecast"
        
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.HTTPError(
                f"Nolano API request failed: {str(e)}"
            ) from e
        
        # Parse response
        try:
            result = response.json()
            return NolanoForecast(
                forecast_timestamps=result['forecast_timestamps'],
                lower_bound=result['lower_bound'],
                median=result['median'],
                upper_bound=result['upper_bound']
            )
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid response format from Nolano API: {str(e)}") from e
    
    def forecast_from_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        forecast_horizon: int,
        data_frequency: str,
        forecast_frequency: Optional[str] = None,
        confidence: float = 0.95,
        model_id: Optional[str] = None
    ) -> NolanoForecast:
        """Generate forecast from pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with time series data
            timestamp_col (str): Column name containing timestamps
            value_col (str): Column name containing values to forecast
            forecast_horizon (int): Number of future periods to predict
            data_frequency (str): Frequency of input data
            forecast_frequency (str, optional): Desired forecast frequency. 
                Defaults to data_frequency
            confidence (float): Confidence level. Defaults to 0.95
            model_id (str, optional): Override default model ID
            
        Returns:
            NolanoForecast: Forecast results
            
        Raises:
            KeyError: If specified columns not found in DataFrame
        """
        if timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        
        if value_col not in df.columns:
            raise KeyError(f"Value column '{value_col}' not found in DataFrame")
        
        # Convert DataFrame to Nolano format
        df_sorted = df.sort_values(timestamp_col)
        
        # Convert timestamps to required format
        timestamps = pd.to_datetime(df_sorted[timestamp_col])
        timestamp_strings = timestamps.dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        
        values = df_sorted[value_col].astype(float).tolist()
        
        series = [{
            'timestamps': timestamp_strings,
            'values': values
        }]
        
        return self.forecast(
            series=series,
            forecast_horizon=forecast_horizon,
            data_frequency=data_frequency,
            forecast_frequency=forecast_frequency or data_frequency,
            confidence=confidence,
            model_id=model_id
        ) 
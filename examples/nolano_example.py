"""
Nolano Python SDK Example

This script demonstrates how to use the Nolano Python SDK for time series forecasting.

Prerequisites:
1. Install the Nolano package: pip install nolano
2. Get a Nolano API key from https://api.nolano.ai
3. Set your API key as environment variable or replace the placeholder below
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import Nolano SDK
from nolano import Nolano, NolanoClient, NolanoForecast


def create_sample_data():
    """Create sample time series data for demonstration."""
    print("Creating sample time series data...")
    
    # Create sample daily sales data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Generate synthetic sales data with trend and seasonality
    np.random.seed(42)
    trend = np.linspace(100, 150, 365)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 10, 365)
    sales = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    print(f"✓ Created dataset with {len(df)} data points")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Sales range: {df['sales'].min():.2f} to {df['sales'].max():.2f}")
    
    return df


def demo_basic_forecasting():
    """Demonstrate basic Nolano forecasting functionality."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Nolano Forecasting")
    print("="*60)
    
    # Create sample data
    df = create_sample_data()
    
    # Initialize Nolano client
    # Replace with your actual API key or set NOLANO_API_KEY environment variable
    try:
        client = Nolano(api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"))
    except ValueError as e:
        print(f"  Error: {e}")
        print("  Please set NOLANO_API_KEY environment variable or provide api_key")
        return None, None
    
    # List available Nolano models
    print(f"\nCurrent model: {client.model_id}")
    print("\nAvailable Nolano models:")
    try:
        models = client.list_models()
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    except Exception as e:
        print(f"  Error listing models: {e}")
        return df, None
    
    # Validate data before forecasting
    print("\nValidating data...")
    validation = client.validate_data(df, 'sales', 'date')
    if validation['valid']:
        print("✓ Data validation passed")
        print(f"  Total data points: {validation['stats']['total_rows']}")
        print(f"  Date range: {validation['stats']['date_range']['start']} to {validation['stats']['date_range']['end']}")
    else:
        print("✗ Data validation failed:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Generate forecast using Nolano API
    print("\nGenerating 30-day forecast...")
    try:
        forecast = client.forecast(
            dataset=df,
            target_col='sales',
            timestamp_col='date',
            forecast_horizon=30,
            data_frequency='Daily',
            confidence=0.95
        )
        
        print(f"✓ Forecast generated successfully!")
        print(f"  Forecast horizon: {len(forecast.median)} days")
        print(f"  Average forecast: {np.mean(forecast.median):.2f}")
        print(f"  Forecast range: {min(forecast.median):.2f} to {max(forecast.median):.2f}")
        
        # Convert to DataFrame
        forecast_df = forecast.to_dataframe()
        print(f"  Forecast DataFrame shape: {forecast_df.shape}")
        
        return df, forecast
        
    except Exception as e:
        print(f"✗ Error generating forecast: {e}")
        return df, None


def demo_direct_client():
    """Demonstrate using NolanoClient directly."""
    print("\n" + "="*60)
    print("DEMO 2: Direct Nolano Client Usage")
    print("="*60)
    
    df = create_sample_data()
    
    # Initialize Nolano client directly
    try:
        nolano_client = NolanoClient(
            api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"),
            model_id="forecast-model-2"  # Use alternative model as default
        )
        
        print(f"✓ Direct client initialized with model: {nolano_client.model_id}")
        
        # Generate forecast using direct client
        print("\nGenerating forecast with direct client...")
        forecast = nolano_client.forecast_from_dataframe(
            df=df,
            timestamp_col='date',
            value_col='sales',
            forecast_horizon=14,  # Two weeks
            data_frequency='Daily',
            confidence=0.80
        )
        
        print(f"✓ Direct forecast generated!")
        print(f"  Forecast length: {len(forecast.median)} days")
        print(f"  80% confidence interval width: {np.mean(np.array(forecast.upper_bound) - np.array(forecast.lower_bound)):.2f}")
        
        return forecast
        
    except Exception as e:
        print(f"✗ Error with direct client: {e}")
        return None


def demo_model_comparison():
    """Demonstrate comparing different Nolano models."""
    print("\n" + "="*60)
    print("DEMO 3: Model Comparison")
    print("="*60)
    
    df = create_sample_data()
    
    try:
        client = Nolano(api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"))
        
        # Compare first two models
        models_to_compare = ['forecast-model-1', 'forecast-model-2']
        forecasts = {}
        
        for model_id in models_to_compare:
            try:
                print(f"\nGenerating forecast with {model_id}...")
                
                # Get model information
                model_info = client.get_model_info(model_id)
                print(f"  Model: {model_info['name']}")
                print(f"  Use cases: {model_info['use_cases']}")
                
                forecast = client.forecast(
                    dataset=df,
                    target_col='sales',
                    timestamp_col='date',
                    forecast_horizon=7,  # One week for comparison
                    data_frequency='Daily',
                    model_id=model_id
                )
                forecasts[model_id] = forecast
                print(f"✓ {model_id}: avg forecast = {np.mean(forecast.median):.2f}")
                
            except Exception as e:
                print(f"✗ {model_id}: error = {e}")
        
        # Compare results
        if len(forecasts) >= 2:
            print("\nModel comparison:")
            for model_id, forecast in forecasts.items():
                avg_forecast = np.mean(forecast.median)
                std_forecast = np.std(forecast.median)
                print(f"  {model_id}: mean={avg_forecast:.2f}, std={std_forecast:.2f}")
        
        return forecasts
        
    except Exception as e:
        print(f"✗ Error in model comparison: {e}")
        return {}


def demo_frequency_handling():
    """Demonstrate different time series frequencies."""
    print("\n" + "="*60)
    print("DEMO 4: Frequency Handling")
    print("="*60)
    
    # Create hourly data for demonstration
    start_hour = datetime(2024, 1, 1)
    hours = [start_hour + timedelta(hours=i) for i in range(168)]  # 1 week of hourly data
    
    # Generate synthetic hourly sales with daily pattern
    np.random.seed(42)
    hourly_pattern = 50 + 30 * np.sin(2 * np.pi * np.arange(168) / 24)  # Daily pattern
    hourly_noise = np.random.normal(0, 5, 168)
    hourly_sales = hourly_pattern + hourly_noise
    
    hourly_df = pd.DataFrame({
        'hour': hours,
        'sales': hourly_sales
    })
    
    print(f"Created hourly dataset: {hourly_df.shape}")
    
    try:
        client = Nolano(api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"))
        
        # Generate hourly forecast
        print("\nGenerating 24-hour forecast...")
        hourly_forecast = client.forecast(
            dataset=hourly_df,
            target_col='sales',
            timestamp_col='hour',
            forecast_horizon=24,  # Next 24 hours
            data_frequency='Hours',
            confidence=0.90
        )
        
        print(f"✓ Hourly forecast generated for {len(hourly_forecast.median)} hours")
        print(f"  Average hourly forecast: {np.mean(hourly_forecast.median):.2f}")
        
        return hourly_forecast
        
    except Exception as e:
        print(f"✗ Error with hourly forecasting: {e}")
        return None


def demo_data_conversion():
    """Demonstrate data format conversion utilities."""
    print("\n" + "="*60)
    print("DEMO 5: Data Conversion & Raw Series")
    print("="*60)
    
    # Create small sample dataset
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'value': [100, 102, 98, 105, 103, 107, 99, 104, 106, 101]
    })
    
    print("Sample DataFrame:")
    print(sample_df)
    
    try:
        client = Nolano(api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"))
        
        # Convert DataFrame to raw series format
        from nolano.utils import forecast_to_nolano_format
        
        nolano_series = forecast_to_nolano_format(
            df=sample_df,
            timestamp_col='timestamp',
            value_col='value'
        )
        
        print("\nNolano series format:")
        print(f"  Timestamps: {nolano_series['timestamps'][:3]}...")
        print(f"  Values: {nolano_series['values'][:3]}...")
        
        # Use raw series format for forecasting
        print("\nForecasting with raw series data...")
        forecast = client.forecast_from_series(
            series=[nolano_series],
            forecast_horizon=5,
            data_frequency='Daily'
        )
        
        print(f"✓ Raw series forecast generated for {len(forecast.median)} days")
        
    except Exception as e:
        print(f"✗ Error with data conversion: {e}")


def demo_error_handling():
    """Demonstrate proper error handling."""
    print("\n" + "="*60)
    print("DEMO 6: Error Handling Examples")
    print("="*60)
    
    df = create_sample_data()
    
    # Test 1: Invalid API key
    print("1. Testing invalid API key:")
    try:
        client = Nolano(api_key="invalid_api_key")
        forecast = client.forecast(
            dataset=df,
            target_col='sales',
            timestamp_col='date',
            forecast_horizon=5,
            data_frequency='Daily'
        )
    except Exception as e:
        print(f"   ✓ Expected error caught: {type(e).__name__}")
    
    # Test 2: Invalid column names
    print("\n2. Testing invalid column names:")
    try:
        client = Nolano(api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"))
        forecast = client.forecast(
            dataset=df,
            target_col='invalid_column',  # Invalid column
            timestamp_col='date',
            forecast_horizon=5,
            data_frequency='Daily'
        )
    except KeyError as e:
        print(f"   ✓ Expected error caught: {e}")
    
    # Test 3: Invalid confidence level
    print("\n3. Testing invalid confidence level:")
    try:
        client = Nolano(api_key=os.getenv("NOLANO_API_KEY", "your_nolano_api_key"))
        forecast = client.forecast(
            dataset=df,
            target_col='sales',
            timestamp_col='date',
            forecast_horizon=5,
            data_frequency='Daily',
            confidence=1.5  # Invalid confidence > 1
        )
    except ValueError as e:
        print(f"   ✓ Expected error caught: {e}")
    
    print("\n✓ Error handling demonstrations complete")


def main():
    """Run all demonstrations."""
    print("Nolano Python SDK Demonstration")
    print("=" * 80)
    
    print("\nNote: Make sure to set your API key:")
    print("  export NOLANO_API_KEY='your_nolano_api_key'")
    print("\nOr modify the script to include your key directly.")
    
    # Run demonstrations
    try:
        # Basic forecasting
        df, forecast = demo_basic_forecasting()
        
        # Direct client usage
        demo_direct_client()
        
        # Model comparison
        forecasts = demo_model_comparison()
        
        # Frequency handling
        demo_frequency_handling()
        
        # Data conversion
        demo_data_conversion()
        
        # Error handling
        demo_error_handling()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nKey takeaways:")
        print("1. ✓ Multiple specialized models for different use cases")
        print("2. ✓ Simple API for forecasting from pandas DataFrames")
        print("3. ✓ Support for various time series frequencies")
        print("4. ✓ Built-in data validation and error handling")
        print("5. ✓ Confidence intervals for uncertainty quantification")
        print("6. ✓ Both high-level and direct client access patterns")
        
        # Show basic visualization if matplotlib available and data exists
        if forecast is not None:
            try:
                print("\nGenerating visualization...")
                plt.figure(figsize=(15, 8))
                
                # Plot last 60 days of historical data
                recent_data = df.tail(60)
                plt.plot(recent_data['date'], recent_data['sales'], 
                        label='Historical Sales', color='blue', linewidth=2)
                
                # Plot forecast
                forecast_df = forecast.to_dataframe()
                plt.plot(forecast_df['timestamp'], forecast_df['median'], 
                        label='Nolano Forecast', color='red', linewidth=2)
                
                # Plot confidence interval
                plt.fill_between(forecast_df['timestamp'], 
                               forecast_df['lower_bound'], 
                               forecast_df['upper_bound'],
                               alpha=0.3, color='red', label='95% Confidence')
                
                plt.axvline(x=df['date'].iloc[-1], color='gray', 
                           linestyle='--', alpha=0.7, label='Forecast Start')
                
                plt.title('Sales Forecast using Nolano API', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Sales', fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                print("✓ Visualization complete")
                
            except Exception as e:
                print(f"Note: Could not generate visualization: {e}")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Check your API key and internet connection.")


if __name__ == "__main__":
    main() 
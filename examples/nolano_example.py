#!/usr/bin/env python3

"""
Example usage of Nolano SDK for time series forecasting.

This example demonstrates:
1. API key verification
2. Data preparation and validation
3. Generating forecasts with different models
4. Working with forecast results
5. Using built-in plotting functionality

For real-world dataset examples, see real_data_example.py which uses:
- Walmart Sales Data
- Corporacion Favorita Store Data
"""

import pandas as pd
import numpy as np
from nolano import Nolano

def create_sample_data():
    """Create sample time series data for demonstration."""
    print("\n📊 Creating Sample Data")
    print("=" * 25)
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sales = np.linspace(100, 150, 100)
    
    df = pd.DataFrame({'date': dates, 'sales': sales})
    
    print(f"Created {len(df)} data points")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sales range: {df['sales'].min():.2f} to {df['sales'].max():.2f}")
    
    return df

def test_single_model(client, model_id, df):
    """Test a single model and return results."""
    print(f"\nTesting {model_id}...")
    
    try:
        info = client.get_model_info(model_id)
        print(f"   Name: {info['name']}")
        print(f"   Use cases: {info['use_cases']}")
        
        forecast = client.forecast(
            dataset=df,
            target_col='sales',
            timestamp_col='date',
            forecast_horizon=14,
            data_frequency='Daily',
            model_id=model_id
        )
        
        forecast_df = forecast.to_dataframe()
        
        # Use the built-in plot method - demonstrate both display and save functionality
        print(f"   📊 Displaying forecast plot for {model_id}...")
        try:
            # Display the plot
            forecast.plot(height=5, width=8)
            
            # Save the plot to a file
            save_filename = f"forecast_{model_id.replace('-', '_')}.png"
            forecast.plot(height=5, width=8, save_path=save_filename)
            print(f"   💾 Plot saved as: {save_filename}")
            
        except Exception as plot_error:
            print(f"   ⚠️ Plotting failed: {plot_error}")
        
        print(f"   ✅ Forecast generated successfully")
        
        return forecast
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return None

def model_comparison_example(client, df):
    """Compare different Nolano models using built-in plotting."""
    print("\n🔄 Model Comparison Example")
    print("=" * 30)
    
    models = client.list_models()
    print("Available models:")
    for model in models:
        print(f"   - {model}")
    
    test_models = ['forecast-model-1', 'forecast-model-2', 'forecast-model-3', 'forecast-model-4']
    successful_forecasts = []
    
    for model_id in test_models:
        forecast = test_single_model(client, model_id, df)
        if forecast is not None:
            successful_forecasts.append((model_id, forecast))
    
    if successful_forecasts:
        print(f"\n📊 Model Comparison Summary:")
        print("=" * 35)
        for model_id, forecast in successful_forecasts:
            print(f"{model_id}:")
            print(f"   ✅ Forecast generated and plotted successfully")
    
    return successful_forecasts

def main():
    """Main example function."""
    print("Nolano SDK Example")
    print("=" * 18)
    
    # Initialize client and verify API key
    client = Nolano()
    print("Verifying API key...")
    result = client.verify_api_key()
    print(result)
    if result['valid']==True:
        print(f"✅ Success: {result['message']}")
        
        # Create sample data and run model comparison
        df = create_sample_data()
        forecasts = model_comparison_example(client, df)
        print("\n🎉 Example completed!")
    else:
        print(f"❌ Failed: {result['message']}")
        print("\n⚠️ Cannot continue without valid API key")

if __name__ == "__main__":
    main() 
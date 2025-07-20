#!/usr/bin/env python3

"""
Example usage of Nolano SDK for time series forecasting.

This example demonstrates:
1. API key verification
2. Data preparation and validation
3. Generating forecasts with different models
4. Working with forecast results
5. Plotting forecast results with matplotlib
6. Comparing 4 different models side-by-side in a single plot

For real-world dataset examples, see real_data_example.py which uses:
- Walmart Sales Data
- Corporacion Favorita Store Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

def plot_model_comparison(historical_df, model_forecasts, target_col='sales', timestamp_col='date'):
    """Plot model comparison with subplots for each model."""
    print("\n📊 Plotting Model Comparison")
    print("=" * 30)
    
    # Create subplots for each model
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (model_id, forecast_df) in enumerate(model_forecasts.items()):
        if i >= 4:  # Safety check in case we have more than 4 models
            break
            
        ax = axes[i]
        color = colors[i % len(colors)]
        
        # Plot historical data
        ax.plot(historical_df[timestamp_col], historical_df[target_col], 
                label='Historical Data', color='black', linewidth=2, alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast_df['timestamp'], forecast_df['median'], 
                label=f'{model_id} Forecast', color=color, linewidth=2, 
                linestyle='--', alpha=0.9)
        
        # Plot confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            ax.fill_between(forecast_df['timestamp'], 
                           forecast_df['lower_bound'], 
                           forecast_df['upper_bound'],
                           alpha=0.2, color=color, 
                           label=f'{model_id} Confidence')
        
        # Customize subplot
        ax.set_title(f'{model_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Sales', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis='x', rotation=45)
    
    # Add overall title
    fig.suptitle('Model Comparison: Time Series Forecasts', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('model_comparison_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Model comparison plot saved: model_comparison.png")

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
        results = {
            'forecast_df': forecast_df
        }
        
        print(f"   ✅ Forecast generated successfully")
        
        return results, forecast_df
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return None, None

def model_comparison_example(client, df):
    """Compare different Nolano models."""
    print("\n🔄 Model Comparison Example")
    print("=" * 30)
    
    models = client.list_models()
    print("Available models:")
    for model in models:
        print(f"   - {model}")
    
    model_results = {}
    model_forecasts = {}
    test_models = ['forecast-model-1', 'forecast-model-2', 'forecast-model-3', 'forecast-model-4']
    
    for model_id in test_models:
        results, forecast_df = test_single_model(client, model_id, df)
        if results is not None and forecast_df is not None:
            model_results[model_id] = results
            model_forecasts[model_id] = forecast_df
    
    if model_forecasts:
        plot_model_comparison(df, model_forecasts)
        
        print(f"\n📊 Model Comparison Summary:")
        print("=" * 35)
        for model_id, results in model_results.items():
            print(f"{model_id}:")
            print(f"   ✅ Forecast generated successfully")
    
    return model_results

def main():
    """Main example function."""
    print("Nolano SDK Example")
    print("=" * 18)
    
    # Initialize client and verify API key
    client = Nolano()
    print("Verifying API key...")
    result = client.verify_api_key()
    
    if result['valid']:
        print(f"✅ Success: {result['message']}")
        print(f"   Status: {result['status']}")
        
        # Create sample data and run model comparison
        df = create_sample_data()
        model_results = model_comparison_example(client, df)
        print("\n🎉 Example completed!")
    else:
        print(f"❌ Failed: {result['message']}")
        print("\n⚠️ Cannot continue without valid API key")

if __name__ == "__main__":
    main() 
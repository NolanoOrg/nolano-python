# Nolano Python SDK

<p align="center">
  <a href="https://app.nolano.ai">
    <img src="https://img.shields.io/badge/API-Nolano-blue" alt="nolano_badge">
  </a>
  <a href="https://pypi.org/project/nolano/">
    <img src="https://img.shields.io/pypi/v/nolano.svg" alt="PyPI Badge">
  </a>
</p>

### Nolano - Advanced Time Series Forecasting API

The Nolano Python SDK provides easy access to Nolano's powerful time series forecasting API, featuring multiple specialized models for different forecasting scenarios. Get accurate predictions with minimal setup using our REST API and Python client.

<p align="center">
    <a href="https://docs.nolano.ai">API Documentation</a>
    ·
    <a href="https://github.com/nolano/nolano-python/issues/new">Report Bug</a>
    ·
    <a href="https://github.com/nolano/nolano-python">GitHub</a>
</p>

## 🔥 Features

* **Multiple Forecasting Models**: Choose from 4 specialized models optimized for different use cases
* **Simple API**: Clean, intuitive Python interface for time series forecasting
* **Flexible Input**: Support for pandas DataFrames and raw time series data
* **Confidence Intervals**: Configurable prediction intervals for uncertainty quantification
* **Multiple Frequencies**: Support for various time series frequencies (daily, hourly, etc.)
* **Data Validation**: Built-in validation and helpful error messages

## 🚀 Getting Started

To use the Nolano SDK, you'll need an API key from Nolano:

1. Visit [https://app.nolano.ai](https://app.nolano.ai)
2. Sign up for an account
3. Generate an API key
4. Set your API key as an environment variable or pass it directly

## ⚙️ Installation

Install the Nolano SDK using pip:

```bash
pip install nolano
```

## Quick Start Example

```python
from nolano import Nolano
import pandas as pd

# Initialize the client
client = Nolano(api_key="your_api_key_here")

# Or set environment variable: NOLANO_API_KEY=your_api_key_here
client = Nolano()

# Verify API key (recommended)
verification = client.verify_api_key()
if not verification['valid']:
    print(f"API key issue: {verification['message']}")
    exit(1)

print("✅ API key verified successfully!")

# Prepare your time series data
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'sales': [100, 102, 98, 105, ...]  # Your time series values
})

# Generate forecast
forecast = client.forecast(
    dataset=df,
    target_col='sales',
    timestamp_col='date',
    forecast_horizon=30,
    data_frequency='Daily'
)

# Access results
print(f"Forecast: {forecast.median}")
print(f"Lower bound: {forecast.lower_bound}")
print(f"Upper bound: {forecast.upper_bound}")

# Convert to DataFrame for analysis
forecast_df = forecast.to_dataframe()
```

## 📊 Available Models

The Nolano API provides four specialized forecasting models:

| Model ID | Description | Best For |
|----------|-------------|----------|
| `forecast-model-1` | **Primary Forecasting Model** | General forecasting tasks, trend analysis, business planning |
| `forecast-model-2` | **Alternative Forecasting Model** | Complex seasonal patterns, anomaly detection |
| `forecast-model-3` | **Advanced Forecasting Model** | High-frequency data, real-time forecasting, large datasets |
| `forecast-model-4` | **Next-Generation Model** | Complex patterns, ensemble predictions |

```python
# List available models
models = client.list_models()
for model in models:
    print(model)

# Use a specific model
forecast = client.forecast(
    dataset=df,
    target_col='sales',
    timestamp_col='date',
    forecast_horizon=30,
    data_frequency='Daily',
    model_id='forecast-model-2'  # Use alternative model
)
```

## 🔧 API Reference

### Nolano Class

The main class for interacting with the Nolano API.

```python
client = Nolano(
    api_key="your_api_key",      # Optional if NOLANO_API_KEY env var is set
    model_id="forecast-model-1"  # Default model to use
)
```

### verify_api_key()

Verify that your API key is valid and has the necessary permissions.

```python
# Verify API key
result = client.verify_api_key()

if result['valid']:
    print("✅ API key is valid!")
    print(f"Status: {result['status']}")
else:
    print(f"❌ API key issue: {result['message']}")
    print(f"Status: {result['status']}")
```

**Response Format:**
```python
{
    'valid': bool,          # True if API key is valid
    'status': str,          # 'success', 'unauthorized', or 'forbidden'  
    'message': str          # Human-readable status message
}
```

### forecast()

Generate time series forecasts from a pandas DataFrame.

```python
forecast = client.forecast(
    dataset=df,                    # pandas DataFrame with time series data
    target_col='sales',           # Column name containing values to forecast
    timestamp_col='date',         # Column name containing timestamps  
    forecast_horizon=30,          # Number of periods to forecast
    data_frequency='Daily',       # Data frequency: Daily, Hourly, Weekly, etc.
    forecast_frequency='Daily',   # Forecast frequency (optional, defaults to data_frequency)
    confidence=0.95,             # Confidence level for prediction intervals
    model_id='forecast-model-1'  # Model to use (optional, uses default)
)
```

### forecast_from_series()

Generate forecasts from raw time series data.

```python
series_data = [{
    'timestamps': ['2023-01-01T00:00:00', '2023-01-02T00:00:00', ...],
    'values': [100, 102, 98, ...]
}]

forecast = client.forecast_from_series(
    series=series_data,
    forecast_horizon=30,
    data_frequency='Daily'
)
```

### NolanoForecast Class

The forecast result object with prediction data.

```python
# Access forecast data
forecast.forecast_timestamps  # List of forecast timestamp strings
forecast.median              # List of median forecast values
forecast.lower_bound         # List of lower confidence bound values
forecast.upper_bound         # List of upper confidence bound values

# Convert to DataFrame
df = forecast.to_dataframe()
```

## 📈 Supported Frequencies

The Nolano API supports the following time series frequencies:

- `Seconds` - Second-level data
- `Minutes` - Minute-level data  
- `Hours` - Hourly data
- `Daily` - Daily data
- `Weekly` - Weekly data
- `Monthly` - Monthly data
- `Quarterly` - Quarterly data
- `Yearly` - Annual data

## 🛠️ Advanced Usage

### Data Validation

Validate your data before forecasting:

```python
validation = client.validate_data(
    dataset=df,
    target_col='sales',
    timestamp_col='date'
)

if validation['valid']:
    print("Data is valid!")
    print(f"Stats: {validation['stats']}")
else:
    print("Data issues found:")
    for warning in validation['warnings']:
        print(f"- {warning}")
```

### Model Information

Get detailed information about models:

```python
info = client.get_model_info('forecast-model-1')
print(f"Name: {info['name']}")
print(f"Description: {info['description']}")
print(f"Use cases: {info['use_cases']}")
```

### Direct Client Access

For advanced usage, access the underlying client:

```python
nolano_client = client.get_client()
# Use NolanoClient methods directly
```

## 🔄 Error Handling

The SDK provides helpful error messages for common issues:

```python
# Verify API key before making requests
try:
    result = client.verify_api_key()
    if not result['valid']:
        print(f"API key verification failed: {result['message']}")
        # Handle invalid API key case
        exit(1)
    
    print("API key verified successfully!")
    
    # Proceed with forecasting
    forecast = client.forecast(
        dataset=df,
        target_col='sales',
        timestamp_col='date',
        forecast_horizon=30,
        data_frequency='Daily'
    )
    
except ValueError as e:
    print(f"Parameter error: {e}")
except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"API error: {e}")
```

**Common verification scenarios:**

```python
# Check API key on initialization
client = Nolano(api_key="your_api_key")
verification = client.verify_api_key()

if verification['status'] == 'unauthorized':
    print("Invalid API key - please check your credentials")
elif verification['status'] == 'forbidden':
    print("API key lacks required permissions")
elif verification['status'] == 'success':
    print("API key verified - ready to forecast!")
```

## 📚 Examples

Check out the examples directory for complete usage examples:

- `examples/verify_api_key.py` - Quick API key verification script
- `examples/nolano_example.py` - Comprehensive usage examples with API verification
- `examples/nolano-forecasting-example.ipynb` - Jupyter notebook tutorial

### Quick API Key Test

To quickly verify your API key is working:

```bash
# Set your API key
export NOLANO_API_KEY=your_api_key_here

# Run verification script
cd examples
python verify_api_key.py
```

Or test directly in Python:

```python
from nolano import Nolano

client = Nolano()
result = client.verify_api_key()

if result['valid']:
    print("✅ API key is valid!")
else:
    print(f"❌ Issue: {result['message']}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [Nolano API Documentation](https://api.nolano.ai)
- [GitHub Repository](https://github.com/nolano/nolano-python)
- [PyPI Package](https://pypi.org/project/nolano/)

## 💬 Support

For support and questions:
- Open an issue on [GitHub](https://github.com/nolano/nolano-python/issues)
- Contact: [team@nolano.ai](mailto:team@nolano.ai)

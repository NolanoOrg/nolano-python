# Nolano SDK Examples

This directory contains examples demonstrating how to use the Nolano SDK for time series forecasting.

## Examples

### 1. Basic Example (`nolano_example.py`)
A simple example using synthetic data that demonstrates:
- API key verification
- Basic data preparation
- Generating forecasts with different models
- Plotting results
- Model comparison

**Usage:**
```bash
python nolano_example.py
```

### 2. Real Data Example (`real_data_example.py`)
A comprehensive example using real-world datasets that demonstrates:
- Loading and preprocessing real datasets
- Data validation and cleaning
- Advanced forecasting with multiple models
- Detailed visualization and analysis
- Model comparison on real data

**Datasets used:**
- **Walmart Sales Data**: Store sales data with holiday indicators (421,571 records)
- **Corporacion Favorita Store Data**: Product sales data (59,046 records)

**Usage:**
```bash
python real_data_example.py
```

### 3. Data Processing Demo (`demo_data_processing.py`)
A demo script that doesn't require an API key and shows:
- Data loading and preprocessing
- Statistical analysis and seasonality detection
- Data visualization and validation
- Preparation for forecasting

**Usage:**
```bash
python demo_data_processing.py
```

## Dataset Information

### Walmart Sales Data
- **Source**: `datasets/wallmart-sales/data.csv`
- **Records**: 421,571
- **Columns**: Store, Dept, Date, Weekly_Sales, IsHoliday
- **Date Range**: 2010-02-05 to 2012-10-26
- **Features**: 
  - 45 stores
  - 81 departments
  - Holiday indicators
  - Weekly sales data

### Corporacion Favorita Store Data
- **Source**: `datasets/corporacion-favorita-store/data.csv`
- **Records**: 59,046
- **Columns**: unique_id, ds, y
- **Date Range**: 2013-01-02 to 2017-08-15
- **Features**:
  - Product-level sales data
  - Daily frequency
  - Multiple products aggregated

## Output Files

Both examples generate several visualization files:

### Basic Example Outputs:
- `model_comparison_graph.png`: Side-by-side comparison of 4 models

### Real Data Example Outputs:
- `walmart_dataset_overview.png`: Walmart data analysis plots
- `corporacion_dataset_overview.png`: Corporacion Favorita data analysis plots
- `walmart_sales_forecast_comparison.png`: Walmart forecast comparisons
- `corporacion_favorita_sales_forecast_comparison.png`: Corporacion Favorita forecast comparisons

### Demo Outputs:
- `walmart_dataset_overview.png`: Walmart data analysis plots
- `corporacion_dataset_overview.png`: Corporacion Favorita data analysis plots

## Prerequisites

1. **API Key**: You need a valid Nolano API key set as an environment variable:
   ```bash
   export NOLANO_API_KEY="your_api_key_here"
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install pandas numpy matplotlib nolano
   ```

## Key Features Demonstrated

### Data Processing
- Loading CSV files with pandas
- Date parsing and validation
- Data aggregation (store/department level → daily totals)
- Data cleaning and preprocessing

### Forecasting
- Multiple model testing
- Configurable forecast horizons
- Different data frequencies (Daily/Weekly)
- Error handling and validation

### Visualization
- Time series plots
- Distribution analysis
- Model comparison charts
- Confidence intervals
- Professional formatting

### Analysis
- Model performance comparison
- Dataset statistics
- Forecast accuracy assessment
- Comprehensive reporting

## Tips for Using Real Data

1. **Data Size**: The real datasets are large, so processing may take time
2. **Memory Usage**: Consider using data sampling for initial testing
3. **API Limits**: Be aware of API rate limits when testing multiple models
4. **Forecast Horizon**: Adjust based on your business needs (8 weeks for weekly data, 30 days for daily data)

## Troubleshooting

- **API Key Issues**: Ensure your API key is valid and properly set
- **Memory Issues**: For large datasets, consider processing in chunks
- **Plot Issues**: Ensure matplotlib backend is properly configured
- **Data Issues**: Check CSV format and encoding if loading fails 
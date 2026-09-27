# WildFireAI Dataset Documentation

This directory contains the dataset used for training and evaluating the WildFireAI Forest Fire Risk Prediction Model.

## Data Source
- **Primary Reference**: Inspired by NASA FIRMS (Fire Information for Resource Management System) MODIS / VIIRS active fire observations and Canadian Forest Fire Weather Index (FWI) System metrics.
- **Sample Dataset**: `sample_data.csv` contains a synthetic, clearly labeled demonstration dataset simulating environmental conditions (temperature, humidity, wind speed, precipitation, vegetation dryness) and active fire occurrences.

## Features
- `latitude`: Geographical latitude coordinates (-90 to 90)
- `longitude`: Geographical longitude coordinates (-180 to 180)
- `region`: Region/zone identifier (e.g. Northern Forest, Central Valley, Coastal Hills, Southern Pine, Eastern Ridge)
- `temperature_c`: Ambient air temperature in Celsius (°C)
- `humidity_pct`: Relative humidity percentage (%)
- `wind_speed_kmh`: Wind speed in kilometers per hour (km/h)
- `rainfall_mm`: Recent 24-hour rainfall / precipitation in millimeters (mm)
- `vegetation_dryness_index`: Fine Fuel Moisture & Vegetation Dryness Index (0 = saturated, 100 = bone dry)
- `historical_fire_freq`: Count of past fire occurrences in the localized cluster over 5 years
- `fire_occurred`: Binary target variable (1 = Active Fire / High Risk Event, 0 = No Fire Event)

import os
import pandas as pd
import numpy as np

# Feature list used across training and inference
FEATURE_COLS = [
    'temperature_c',
    'humidity_pct',
    'wind_speed_kmh',
    'rainfall_mm',
    'vegetation_dryness_index',
    'latitude',
    'longitude',
    'historical_fire_freq',
    'fire_weather_index_proxy'
]

TARGET_COL = 'fire_occurred'

REGIONS = ['Northern Forest', 'Central Valley', 'Coastal Hills', 'Southern Pine', 'Eastern Ridge']


def generate_sample_dataset(filepath: str, num_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, clearly labelled sample dataset for WildFireAI.
    Uses physics-informed formulas inspired by the Canadian Forest Fire Weather Index (FWI).
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Coordinates centered around fire-prone regions (e.g., Western US / Mediterranean / India forest regions)
    lats = np.random.uniform(28.0, 48.0, num_samples)
    lons = np.random.uniform(-124.0, -75.0, num_samples)
    regions = np.random.choice(REGIONS, size=num_samples)

    # Environmental features
    temp = np.random.normal(loc=28.0, scale=8.0, size=num_samples).clip(5.0, 48.0)
    humidity = np.random.normal(loc=45.0, scale=18.0, size=num_samples).clip(8.0, 95.0)
    wind_speed = np.random.normal(loc=18.0, scale=10.0, size=num_samples).clip(1.0, 65.0)
    rainfall = np.random.exponential(scale=3.5, size=num_samples).clip(0.0, 45.0)
    
    # Dryness Index (higher = drier vegetation)
    dryness = (0.6 * temp - 0.4 * humidity + 0.3 * wind_speed - 1.2 * rainfall + np.random.normal(30, 8, num_samples)).clip(0, 100)
    
    # Historical fire frequency (0 to 15 fires in past 5 yrs)
    hist_freq = np.random.poisson(lam=2.5, size=num_samples).clip(0, 15)

    # Fire Weather Index Proxy calculation (physically sound logic)
    # High temp, low humidity, high wind, low rain, high dryness -> high risk
    fwi_score = (
        0.35 * temp 
        - 0.25 * humidity 
        + 0.20 * wind_speed 
        - 0.40 * rainfall 
        + 0.30 * dryness 
        + 0.15 * hist_freq
    )
    
    # Sigmoid probability of fire occurrence
    prob = 1 / (1 + np.exp(-(fwi_score - 22) / 4.5))
    fire_occurred = (np.random.uniform(0, 1, num_samples) < prob).astype(int)

    df = pd.DataFrame({
        'latitude': np.round(lats, 4),
        'longitude': np.round(lons, 4),
        'region': regions,
        'temperature_c': np.round(temp, 1),
        'humidity_pct': np.round(humidity, 1),
        'wind_speed_kmh': np.round(wind_speed, 1),
        'rainfall_mm': np.round(rainfall, 1),
        'vegetation_dryness_index': np.round(dryness, 1),
        'historical_fire_freq': hist_freq,
        'fire_occurred': fire_occurred
    })

    df.to_csv(filepath, index=False)
    print(f"Sample dataset generated successfully at: {filepath}")
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from CSV file. If file does not exist, auto-generates sample dataset.
    """
    if not os.path.exists(filepath):
        print(f"Data file not found at {filepath}. Generating sample dataset...")
        return generate_sample_dataset(filepath)
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}. Falling back to sample dataset generation...")
        return generate_sample_dataset(filepath)


def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses dataset:
    - Fills missing values
    - Engineers composite feature 'fire_weather_index_proxy'
    - Separates features X and target y
    """
    df_clean = df.copy()

    # Fill numerical missing values with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Feature Engineering: Fire Weather Index Proxy
    if 'fire_weather_index_proxy' not in df_clean.columns:
        df_clean['fire_weather_index_proxy'] = np.round(
            (0.35 * df_clean['temperature_c'] 
             - 0.25 * df_clean['humidity_pct'] 
             + 0.20 * df_clean['wind_speed_kmh'] 
             - 0.40 * df_clean['rainfall_mm'] 
             + 0.30 * df_clean['vegetation_dryness_index'] 
             + 0.15 * df_clean['historical_fire_freq']).clip(0, 100), 2
        )

    # Ensure all required features are present
    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL] if TARGET_COL in df_clean.columns else None

    return df_clean, X, y


if __name__ == '__main__':
    # Test script execution
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    df = load_data(sample_path)
    df_processed, X, y = preprocess_data(df)
    print("Preprocessed X shape:", X.shape)
    print("Target distribution:\n", y.value_counts())

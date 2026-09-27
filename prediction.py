import pandas as pd
import numpy as np

try:
    from src.data_processing import FEATURE_COLS
except ImportError:
    from data_processing import FEATURE_COLS


def get_risk_category(risk_score: float) -> dict:
    """
    Returns risk level label, color, icon, and description based on risk_score (0 - 100%).
    """
    if risk_score < 25.0:
        return {
            'level': 'LOW',
            'color': '#2ecc71',  # Emerald Green
            'badge_bg': '#1e3a29',
            'badge_text': '#abf7b1',
            'icon': '🟢',
            'description': 'Normal environmental conditions. Standard precautionary vigilance recommended.'
        }
    elif risk_score < 50.0:
        return {
            'level': 'MODERATE',
            'color': '#f39c12',  # Amber Yellow
            'badge_bg': '#3d2e0f',
            'badge_text': '#fce8a6',
            'icon': '🟡',
            'description': 'Elevated dryness or wind. Exercise caution with open flames and machinery.'
        }
    elif risk_score < 75.0:
        return {
            'level': 'HIGH',
            'color': '#e67e22',  # Burnt Orange
            'badge_bg': '#42220c',
            'badge_text': '#ffd1aa',
            'icon': '🟠',
            'description': 'Dangerous fire weather conditions. Ban outdoor burning and restrict hot work.'
        }
    else:
        return {
            'level': 'EXTREME',
            'color': '#e74c3c',  # Crimson Red
            'badge_bg': '#4d1519',
            'badge_text': '#ffb5b9',
            'icon': '🔴',
            'description': 'Severe ignition and rapid spread potential. Standby emergency protocols activated.'
        }


def predict_fire_risk(model_bundle: dict, input_params: dict) -> dict:
    """
    Predicts fire risk score (0-100%) and returns detailed risk diagnostics.
    """
    rf_model = model_bundle['model']
    
    # Calculate fire weather index proxy if missing
    temp = float(input_params.get('temperature_c', 25.0))
    humidity = float(input_params.get('humidity_pct', 50.0))
    wind = float(input_params.get('wind_speed_kmh', 15.0))
    rain = float(input_params.get('rainfall_mm', 0.0))
    dryness = float(input_params.get('vegetation_dryness_index', 40.0))
    hist_freq = float(input_params.get('historical_fire_freq', 2.0))
    lat = float(input_params.get('latitude', 37.7749))
    lon = float(input_params.get('longitude', -122.4194))

    fwi_proxy = round(
        (0.35 * temp - 0.25 * humidity + 0.20 * wind - 0.40 * rain + 0.30 * dryness + 0.15 * hist_freq), 2
    )

    input_data = {
        'temperature_c': temp,
        'humidity_pct': humidity,
        'wind_speed_kmh': wind,
        'rainfall_mm': rain,
        'vegetation_dryness_index': dryness,
        'latitude': lat,
        'longitude': lon,
        'historical_fire_freq': hist_freq,
        'fire_weather_index_proxy': fwi_proxy
    }

    input_df = pd.DataFrame([input_data])[FEATURE_COLS]

    # Model probability prediction
    proba = rf_model.predict_proba(input_df)[0][1]
    risk_score = round(proba * 100, 1)

    # Risk level category details
    category_info = get_risk_category(risk_score)

    # Feature contribution breakdown (weighted impact)
    feature_importances = model_bundle.get('feature_importances', {})
    
    # Baseline comparison (normalized risk drivers)
    factors = []
    feature_labels = {
        'temperature_c': 'High Temperature',
        'humidity_pct': 'Low Humidity',
        'wind_speed_kmh': 'High Wind Speed',
        'rainfall_mm': 'Lack of Rain',
        'vegetation_dryness_index': 'Vegetation Dryness',
        'historical_fire_freq': 'Historical Fire History',
        'fire_weather_index_proxy': 'Composite Fire Weather Index',
        'latitude': 'Geographic Latitude',
        'longitude': 'Geographic Longitude'
    }

    # Factor calculation
    raw_drivers = {
        'temperature_c': (temp - 15) / 30,
        'humidity_pct': (80 - humidity) / 70,
        'wind_speed_kmh': (wind - 5) / 40,
        'rainfall_mm': max(0, 10 - rain) / 10,
        'vegetation_dryness_index': dryness / 100,
        'historical_fire_freq': hist_freq / 10,
        'fire_weather_index_proxy': fwi_proxy / 50
    }

    for feat, driver in raw_drivers.items():
        imp = feature_importances.get(feat, 0.1)
        impact_score = round(driver * imp * 100, 2)
        label = feature_labels.get(feat, feat)
        factors.append({'Feature': label, 'Impact': max(0.5, impact_score), 'RawValue': input_data.get(feat)})

    factors_df = pd.DataFrame(factors).sort_values(by='Impact', ascending=True)

    # Actionable safety recommendations
    recommendations = generate_recommendations(risk_score, temp, humidity, wind, rain, dryness)

    return {
        'risk_score': risk_score,
        'category': category_info['level'],
        'color': category_info['color'],
        'badge_bg': category_info['badge_bg'],
        'badge_text': category_info['badge_text'],
        'icon': category_info['icon'],
        'description': category_info['description'],
        'factors': factors_df,
        'input_data': input_data,
        'recommendations': recommendations
    }


def generate_recommendations(score: float, temp: float, humidity: float, wind: float, rain: float, dryness: float) -> list:
    """
    Generates targeted risk recommendations based on score and environmental thresholds.
    """
    recs = []
    
    if score >= 75:
        recs.append("🚨 **Emergency Alert**: Suspend all agricultural, forest clearing, and outdoor burning activities immediately.")
        recs.append("🚒 **Resource Staging**: Position rapid-response firefighting crews on standby in vulnerable sectors.")
    elif score >= 50:
        recs.append("⚠️ **High Vigilance**: Restrict campfire usage and machinery operation near dry vegetation.")
        recs.append("📡 **Active Monitoring**: Increase aerial and satellite watch frequency over forest perimeters.")
    elif score >= 25:
        recs.append("⚡ **Moderate Caution**: Ensure water tankers and hand tools are accessible at field posts.")
        recs.append("📢 **Public Information**: Issue standard wildfire safety notices to rural communities.")
    else:
        recs.append("✅ **Normal Operations**: Continue routine forest monitoring and fuel moisture tracking.")

    # Specific weather triggers
    if wind > 30:
        recs.append(f"💨 **High Wind Warning ({wind} km/h)**: High wind velocities can accelerate fire spread up to 3x faster.")
    if humidity < 25:
        recs.append(f"💧 **Critically Low Humidity ({humidity}%)**: Fine forest fuels ignite rapidly under sub-25% relative humidity.")
    if dryness > 70:
        recs.append(f"🍂 **Extreme Vegetation Dryness ({dryness}/100)**: Vegetation canopy moisture is critically depleted.")
    if rain == 0:
        recs.append("☀️ **Zero Recent Rainfall**: Extended dry spell increases ground litter flammability.")

    return recs

import os
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

# Import custom src modules
from src.data_processing import load_data, preprocess_data, REGIONS
from src.model_training import load_saved_model
from src.prediction import predict_fire_risk
from src.visualization import (
    create_feature_contribution_chart,
    create_feature_importance_chart,
    create_confusion_matrix_chart,
    create_roc_curve_chart,
    create_region_distribution_chart,
    create_environmental_comparison_charts,
    create_risk_distribution_chart,
    create_correlation_heatmap,
    create_historical_trends_chart,
    create_risk_map
)

# Page configuration
st.set_page_config(
    page_title="WildFireAI - Forest Fire Risk Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Environmental-Tech Styling
CUSTOM_CSS = """
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #161b22 0%, #1e2631 100%);
        padding: 24px 32px;
        border-radius: 14px;
        border: 1px solid #232d3b;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        margin-bottom: 24px;
    }
    
    .header-title {
        color: #00d26a;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #8b949e;
        font-size: 16px;
        margin-top: 6px;
    }
    
    /* KPI Cards */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #232d3b;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .kpi-card:hover {
        border-color: #00d26a;
        transform: translateY(-2px);
    }
    
    .kpi-title {
        color: #8b949e;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        color: #e6edf3;
        font-size: 26px;
        font-weight: 700;
        margin: 8px 0 4px 0;
    }
    
    .kpi-sub {
        font-size: 12px;
        color: #00d26a;
    }
    
    /* Prediction Badge */
    .risk-badge {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 30px;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        text-align: center;
        margin-top: 8px;
    }
    
    /* Section Card */
    .content-card {
        background-color: #161b22;
        border: 1px solid #232d3b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #261b12;
        border: 1px solid #e67e22;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffd1aa;
        font-size: 14px;
        margin-top: 20px;
    }
    
    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #12161f;
        border-right: 1px solid #232d3b;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_cached_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'sample_data.csv')
    df = load_data(data_path)
    df_clean, X, y = preprocess_data(df)
    return df_clean


@st.cache_resource(ttl=3600)
def get_cached_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'model.joblib')
    return load_saved_model(model_path)


# Load Data and Model gracefully
try:
    df_data = get_cached_data()
    model_bundle = get_cached_model()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()


# SIDEBAR NAVIGATION
st.sidebar.markdown("### 🌲 WildFireAI Navigation")
st.sidebar.caption("Forest Fire Early Warning System")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "🔥 Fire Risk Predictor",
        "🗺️ Risk Map",
        "📊 Data Analysis",
        "🎯 Model Performance",
        "ℹ️ About / Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 System Status")
st.sidebar.success("🟢 ML Pipeline: Active")
st.sidebar.info("📊 Data Source: NASA FIRMS Sample")
st.sidebar.caption("WildFireAI v1.0.0 | B.Tech Data Science Project")


# 1. DASHBOARD PAGE
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">WILDFIREAI</h1>
        <div style="font-size: 18px; font-weight: 600; color: #00d26a;">Forest Fire Risk Prediction & Early Warning System</div>
        <div class="header-subtitle">Predict environmental fire risk before ignition using machine learning & historical weather analytics.</div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic KPI Calculations from dataset & model
    avg_fwi = df_data['fire_weather_index_proxy'].mean()
    avg_temp = df_data['temperature_c'].mean()
    avg_hum = df_data['humidity_pct'].mean()
    avg_wind = df_data['wind_speed_kmh'].mean()
    avg_rain = df_data['rainfall_mm'].mean()
    high_risk_pct = (df_data['fire_occurred'].sum() / len(df_data)) * 100

    if avg_fwi >= 75:
        dom_level, dom_color = "EXTREME", "#e74c3c"
    elif avg_fwi >= 50:
        dom_level, dom_color = "HIGH", "#e67e22"
    elif avg_fwi >= 25:
        dom_level, dom_color = "MODERATE", "#f39c12"
    else:
        dom_level, dom_color = "LOW", "#2ecc71"

    # KPI Top Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Regional Risk</div>
            <div class="kpi-value" style="color: {dom_color};">{avg_fwi:.1f}%</div>
            <div class="kpi-sub">Avg FWI Proxy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Risk Status</div>
            <div class="kpi-value" style="color: {dom_color}; font-size:20px;">{dom_level}</div>
            <div class="kpi-sub">Dominant Level</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Temp</div>
            <div class="kpi-value">{avg_temp:.1f}°C</div>
            <div class="kpi-sub">Ambient Air</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Humidity</div>
            <div class="kpi-value">{avg_hum:.1f}%</div>
            <div class="kpi-sub">Relative Moisture</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Wind</div>
            <div class="kpi-value">{avg_wind:.1f}</div>
            <div class="kpi-sub">km/h Velocity</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">24h Rainfall</div>
            <div class="kpi-value">{avg_rain:.1f}</div>
            <div class="kpi-sub">mm Precip</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Grid
    row1_left, row1_right = st.columns([1.5, 1])

    with row1_left:
        st.markdown("### 📡 Active Fire Risk Overview")
        st.markdown("Distribution of calculated fire risk indices across monitored regional sectors.")
        st.plotly_chart(create_risk_distribution_chart(df_data), use_container_width=True)

    with row1_right:
        st.markdown("### ⚡ Quick Fire Risk Checker")
        st.markdown("Instant prediction using real-time parameter tuning:")

        q_temp = st.slider("Temperature (°C)", 10.0, 50.0, 34.0, 1.0)
        q_hum = st.slider("Humidity (%)", 5.0, 95.0, 22.0, 1.0)
        q_wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 28.0, 1.0)
        q_rain = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0, 0.5)

        quick_pred = predict_fire_risk(model_bundle, {
            'temperature_c': q_temp,
            'humidity_pct': q_hum,
            'wind_speed_kmh': q_wind,
            'rainfall_mm': q_rain,
            'vegetation_dryness_index': 65.0,
            'historical_fire_freq': 4
        })

        st.markdown(f"""
        <div style="background-color: #161b22; border:1px solid {quick_pred['color']}; padding: 16px; border-radius:10px; text-align:center;">
            <div style="font-size:14px; color:#8b949e;">PREDICTED RISK SCORE</div>
            <div style="font-size:36px; font-weight:800; color:{quick_pred['color']};">{quick_pred['risk_score']}%</div>
            <div class="risk-badge" style="background-color:{quick_pred['badge_bg']}; color:{quick_pred['badge_text']}; border: 1px solid {quick_pred['color']};">
                {quick_pred['icon']} {quick_pred['category']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌲 Regional Fire Risk Breakdown")
    col_reg, col_trend = st.columns([1, 1])

    with col_reg:
        st.plotly_chart(create_region_distribution_chart(df_data), use_container_width=True)
    with col_trend:
        st.plotly_chart(create_historical_trends_chart(df_data), use_container_width=True)


# 2. FIRE RISK PREDICTOR PAGE
elif page == "🔥 Fire Risk Predictor":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🔥 Interactive Fire Risk Predictor</h1>
        <div class="header-subtitle">Enter environmental and location metrics below to calculate machine learning fire ignition probability.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_prediction_form"):
        st.markdown("#### 📍 Location & Region")
        c1, c2, c3 = st.columns(3)
        with c1:
            region = st.selectbox("Region / Sector", REGIONS, index=1)
        with c2:
            lat = st.number_input("Latitude", value=36.7783, format="%.4f")
        with c3:
            lon = st.number_input("Longitude", value=-119.4179, format="%.4f")

        st.markdown("#### 🌡️ Environmental & Weather Parameters")
        e1, e2, e3 = st.columns(3)
        with e1:
            temp = st.slider("Temperature (°C)", min_value=0.0, max_value=55.0, value=35.0, step=0.5)
            humidity = st.slider("Humidity (%)", min_value=5.0, max_value=100.0, value=20.0, step=1.0)
        with e2:
            wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=80.0, value=30.0, step=1.0)
            rainfall = st.number_input("Recent 24h Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        with e3:
            dryness = st.slider("Vegetation Dryness Index (0=Wet, 100=Bone Dry)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            hist_freq = st.number_input("5-Yr Historical Fire Frequency", min_value=0, max_value=20, value=5)

        submit_btn = st.form_submit_button("🔥 PREDICT FIRE RISK", use_container_width=True)

    # Perform Prediction
    input_params = {
        'temperature_c': temp,
        'humidity_pct': humidity,
        'wind_speed_kmh': wind,
        'rainfall_mm': rainfall,
        'vegetation_dryness_index': dryness,
        'historical_fire_freq': hist_freq,
        'latitude': lat,
        'longitude': lon,
        'region': region
    }

    result = predict_fire_risk(model_bundle, input_params)

    # Store prediction in session state for map navigation
    st.session_state['latest_prediction'] = {**input_params, **result}

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📊 Prediction Results")

    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        st.markdown(f"""
        <div style="background-color: #161b22; border: 2px solid {result['color']}; border-radius: 14px; padding: 24px; text-align: center;">
            <div style="font-size:14px; font-weight:700; color:#8b949e; letter-spacing:1px;">FIRE RISK PROBABILITY</div>
            <div style="font-size:56px; font-weight:900; color:{result['color']}; margin: 10px 0;">{result['risk_score']}%</div>
            <div class="risk-badge" style="background-color:{result['badge_bg']}; color:{result['badge_text']}; border: 1px solid {result['color']};">
                {result['icon']} {result['category']} RISK
            </div>
            <p style="margin-top:16px; color:#e6edf3; font-size:14px;">{result['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 🎯 Factors Influencing This Prediction")
        st.markdown("Feature contribution based on model weights and input value deviations:")
        st.plotly_chart(create_feature_contribution_chart(result['factors']), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🛡️ Recommended Early Warning Actions")

    rec_cols = st.columns(2)
    for i, rec in enumerate(result['recommendations']):
        with rec_cols[i % 2]:
            st.info(rec)


# 3. RISK MAP PAGE
elif page == "🗺️ Risk Map":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🗺️ Interactive Wildfire Risk Map</h1>
        <div class="header-subtitle">Geographic visualization of monitored sectors and predicted high-risk zones.</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns([3, 1])

    with m2:
        st.markdown("### 🎛️ Map Controls")
        region_filter = st.multiselect("Filter Regions", REGIONS, default=REGIONS)
        risk_filter = st.slider("Minimum Risk Score Filter (%)", 0, 100, 0)
        
        show_user_point = st.checkbox("Show Latest Form Prediction Target", value=True)

        st.markdown("#### 🎨 Map Legend")
        st.markdown("""
        - 🔴 **Extreme Risk** (75–100%)
        - 🟠 **High Risk** (50–75%)
        - 🟡 **Moderate Risk** (25–50%)
        - 🟢 **Low Risk** (0–25%)
        - 📍 **Target Icon**: User Custom Input
        """)

    with m1:
        # Filter dataset for map
        filtered_df = df_data[
            (df_data['region'].isin(region_filter)) &
            (df_data['fire_weather_index_proxy'] >= risk_filter)
        ]

        pred_point = st.session_state.get('latest_prediction') if show_user_point else None

        folium_map = create_risk_map(filtered_df, predicted_point=pred_point)
        st_folium(folium_map, width="100%", height=550)


# 4. DATA ANALYSIS PAGE (EDA)
elif page == "📊 Data Analysis":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">📊 Exploratory Data Analysis (EDA)</h1>
        <div class="header-subtitle">In-depth statistical analysis of environmental weather variables vs fire occurrences.</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔥 Environmental Impact", "🌡️ Distribution & Correlations", "📈 Trends & Regions"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                create_environmental_comparison_charts(df_data, 'temperature_c', 'Temperature', '°C'),
                use_container_width=True
            )
            st.plotly_chart(
                create_environmental_comparison_charts(df_data, 'wind_speed_kmh', 'Wind Speed', 'km/h'),
                use_container_width=True
            )
        with col_b:
            st.plotly_chart(
                create_environmental_comparison_charts(df_data, 'humidity_pct', 'Relative Humidity', '%'),
                use_container_width=True
            )
            st.plotly_chart(
                create_environmental_comparison_charts(df_data, 'rainfall_mm', 'Recent Rainfall', 'mm'),
                use_container_width=True
            )

    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(create_risk_distribution_chart(df_data), use_container_width=True)
        with col_d:
            st.plotly_chart(create_correlation_heatmap(df_data), use_container_width=True)

    with tab3:
        col_e, col_f = st.columns(2)
        with col_e:
            st.plotly_chart(create_region_distribution_chart(df_data), use_container_width=True)
        with col_f:
            st.plotly_chart(create_historical_trends_chart(df_data), use_container_width=True)


# 5. MODEL PERFORMANCE PAGE
elif page == "🎯 Model Performance":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🎯 Machine Learning Model Performance</h1>
        <div class="header-subtitle">Real-time validation metrics for Random Forest Classifier trained on wild-fire observations.</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = model_bundle['metrics']
    cm = model_bundle['confusion_matrix']
    y_test = model_bundle['y_test']
    y_prob = model_bundle['y_prob_test']
    importances = model_bundle['feature_importances']

    # Performance KPI Cards
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test Accuracy</div>
            <div class="kpi-value" style="color:#00d26a;">{metrics['test_accuracy']:.2%}</div>
            <div class="kpi-sub">Train: {metrics['train_accuracy']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Precision</div>
            <div class="kpi-value">{metrics['precision']:.4f}</div>
            <div class="kpi-sub">Positive Predictive Value</div>
        </div>
        """, unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Recall</div>
            <div class="kpi-value">{metrics['recall']:.4f}</div>
            <div class="kpi-sub">Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)

    with mc4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">F1 Score</div>
            <div class="kpi-value">{metrics['f1_score']:.4f}</div>
            <div class="kpi-sub">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)

    with mc5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi-value">{metrics['roc_auc']:.4f}</div>
            <div class="kpi-sub">Area Under Curve</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(create_confusion_matrix_chart(cm), use_container_width=True)
    with col_m2:
        st.plotly_chart(create_roc_curve_chart(y_test, y_prob, metrics['roc_auc']), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌲 Random Forest Feature Importances")
    st.plotly_chart(create_feature_importance_chart(importances), use_container_width=True)


# 6. ABOUT / METHODOLOGY PAGE
elif page == "ℹ️ About / Methodology":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">ℹ️ About WildFireAI</h1>
        <div class="header-subtitle">2nd-Year B.Tech Data Science Project Architecture & Early Warning Methodology</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Technical Workflow Pipeline
    
    ```
    [ Historical Weather & Satellite Data ]
                    │
                    ▼
          [ Data Processing ] ──► Missing Value Handling & Feature Engineering
                    │
                    ▼
        [ Exploratory Data Analysis ] ──► Weather-to-Fire Correlations
                    │
                    ▼
        [ Random Forest Model ] ──► Supervised Classification & Probability Scoring
                    │
                    ▼
     [ Interactive Risk Dashboard ] ──► 0-100% Risk Category & Map Markers
                    │
                    ▼
      [ Early Warning Alert System ] ──► Actionable Preventive Guidance
    ```

    ---
    
    ### 🌐 Data Source & Attribution
    - **Dataset**: Inspired by **NASA FIRMS (Fire Information for Resource Management System)** MODIS & VIIRS satellite observations combined with Canadian Forest Fire Weather Index (FWI) principles.
    - **Demonstration Mode**: Uses a clearly labeled, physically-consistent sample dataset (`data/sample_data.csv`) ensuring 100% offline functionality without API bottlenecks.

    ---

    ### ⚠️ Mandatory System Disclaimer
    <div class="disclaimer-box">
        <strong>IMPORTANT NOTICE:</strong><br>
        WildFireAI is designed strictly for academic demonstration, environmental research, risk awareness, and preventive decision support. It is <strong>NOT a substitute for official emergency fire detection systems, civil defense authorities, or government emergency services</strong>. Always consult official forestry and emergency services during active fire incidents.
    </div>
    """, unsafe_allow_html=True)


# FOOTER
st.markdown("---")
st.caption("🔥 **WildFireAI Project** | Designed for 2nd-Year B.Tech Data Science Portfolio | Built with Streamlit, Scikit-learn, Plotly & Folium.")

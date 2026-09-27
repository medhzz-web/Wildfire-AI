import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap

# Color Palette for Environmental Tech Theme
THEME_COLORS = {
    'bg': '#0e1117',
    'card_bg': '#161b22',
    'accent_green': '#2ecc71',
    'accent_amber': '#f39c12',
    'accent_orange': '#e67e22',
    'accent_red': '#e74c3c',
    'primary': '#00d26a',
    'text': '#e6edf3',
    'muted': '#8b949e'
}


def apply_dark_theme(fig):
    """Applies a polished dark environmental-tech theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(14, 17, 23, 0.0)',
        plot_bgcolor='rgba(22, 27, 34, 0.4)',
        font=dict(color='#e6edf3', family='Inter, Roboto, sans-serif'),
        title_font=dict(size=16, color='#00d26a', family='Inter, sans-serif'),
        legend=dict(font=dict(color='#e6edf3'), bgcolor='rgba(22,27,34,0.7)'),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.07)', zerolinecolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.07)', zerolinecolor='rgba(255, 255, 255, 0.1)')
    )
    return fig


def create_feature_contribution_chart(factors_df: pd.DataFrame):
    """Bar chart showing factors influencing a specific fire risk prediction."""
    fig = px.bar(
        factors_df,
        x='Impact',
        y='Feature',
        orientation='h',
        title='Factors Influencing Prediction (% Impact Score)',
        color='Impact',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        text='Impact'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
    return apply_dark_theme(fig)


def create_feature_importance_chart(feature_importances: dict):
    """Horizontal bar chart for model overall feature importances."""
    df_imp = pd.DataFrame([
        {'Feature': k.replace('_', ' ').title(), 'Importance': v}
        for k, v in feature_importances.items()
    ]).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Random Forest Model Feature Importances',
        color='Importance',
        color_continuous_scale=['#1e3a29', '#00d26a']
    )
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')
    fig.update_layout(coloraxis_showscale=False)
    return apply_dark_theme(fig)


def create_confusion_matrix_chart(cm_matrix):
    """Plotly heatmap of the confusion matrix."""
    labels = ['No Fire (0)', 'Fire Event (1)']
    fig = px.imshow(
        cm_matrix,
        x=labels,
        y=labels,
        color_continuous_scale=['#161b22', '#1e3a29', '#00d26a'],
        text_auto=True,
        title='Confusion Matrix (Test Evaluation)'
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        coloraxis_showscale=False
    )
    return apply_dark_theme(fig)


def create_roc_curve_chart(y_true, y_prob, roc_auc):
    """Plotly ROC Curve display."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Random Forest (AUC = {roc_auc:.3f})',
        line=dict(color='#00d26a', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Baseline',
        line=dict(color='#8b949e', dash='dash')
    ))
    fig.update_layout(
        title=f'Receiver Operating Characteristic (ROC-AUC = {roc_auc:.3f})',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend=dict(x=0.6, y=0.1)
    )
    return apply_dark_theme(fig)


def create_region_distribution_chart(df: pd.DataFrame):
    """EDA: Fire occurrences count by Region."""
    region_summary = df.groupby(['region', 'fire_occurred']).size().reset_index(name='count')
    region_summary['fire_label'] = region_summary['fire_occurred'].map({0: 'No Fire', 1: 'Fire Event'})

    fig = px.bar(
        region_summary,
        x='region',
        y='count',
        color='fire_label',
        barmode='group',
        title='Fire Occurrences by Region',
        color_discrete_map={'No Fire': '#2ecc71', 'Fire Event': '#e74c3c'}
    )
    fig.update_layout(xaxis_title="Geographic Region", yaxis_title="Sample Observations")
    return apply_dark_theme(fig)


def create_environmental_comparison_charts(df: pd.DataFrame, feature_name: str, label_name: str, unit: str):
    """EDA Box/Violin plot comparing environmental metric vs Fire Event."""
    df_plot = df.copy()
    df_plot['fire_label'] = df_plot['fire_occurred'].map({0: 'No Fire', 1: 'Active Fire'})

    fig = px.box(
        df_plot,
        x='fire_label',
        y=feature_name,
        color='fire_label',
        title=f'{label_name} Distribution by Fire Occurrence',
        color_discrete_map={'No Fire': '#2ecc71', 'Active Fire': '#e74c3c'},
        points="all"
    )
    fig.update_layout(
        xaxis_title="Status",
        yaxis_title=f"{label_name} ({unit})",
        showlegend=False
    )
    return apply_dark_theme(fig)


def create_risk_distribution_chart(df: pd.DataFrame):
    """EDA Histogram of Fire Weather Index Proxy."""
    fig = px.histogram(
        df,
        x='fire_weather_index_proxy',
        color='fire_occurred',
        nbins=30,
        title='Distribution of Fire Weather Index Proxy',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        opacity=0.75,
        barmode='overlay'
    )
    fig.update_layout(xaxis_title="Fire Weather Index Score", yaxis_title="Count")
    return apply_dark_theme(fig)


def create_correlation_heatmap(df: pd.DataFrame):
    """EDA Correlation Heatmap for key features."""
    numeric_cols = [
        'temperature_c', 'humidity_pct', 'wind_speed_kmh',
        'rainfall_mm', 'vegetation_dryness_index', 'historical_fire_freq',
        'fire_weather_index_proxy', 'fire_occurred'
    ]
    corr = df[numeric_cols].corr().round(2)

    labels = [c.replace('_', ' ').title() for c in numeric_cols]

    fig = px.imshow(
        corr,
        x=labels,
        y=labels,
        color_continuous_scale=['#161b22', '#2ecc71', '#e74c3c'],
        text_auto=True,
        title='Environmental Feature Correlation Matrix'
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_dark_theme(fig)


def create_historical_trends_chart(df: pd.DataFrame):
    """EDA Simulated Historical Trend over simulated sample sequence."""
    df_sorted = df.copy().reset_index()
    df_sorted['sample_id'] = df_sorted.index + 1
    df_sorted['moving_fwi'] = df_sorted['fire_weather_index_proxy'].rolling(window=30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['sample_id'], y=df_sorted['fire_weather_index_proxy'],
        mode='lines', name='Sample FWI Score', line=dict(color='#8b949e', width=1), opacity=0.4
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted['sample_id'], y=df_sorted['moving_fwi'],
        mode='lines', name='30-Sample Moving Avg Risk', line=dict(color='#00d26a', width=3)
    ))
    fig.update_layout(
        title='Historical Fire Weather Risk Trend (Sample Timeline)',
        xaxis_title='Observation Timeline',
        yaxis_title='Fire Weather Risk Score'
    )
    return apply_dark_theme(fig)


def create_risk_map(df: pd.DataFrame, predicted_point: dict = None):
    """
    Creates an interactive Folium map centered on the observations with color-coded risk markers.
    """
    if len(df) > 0:
        center_lat = float(df['latitude'].mean())
        center_lon = float(df['longitude'].mean())
    else:
        center_lat, center_lon = 37.7749, -122.4194

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB dark_matter',
        control_scale=True
    )

    marker_cluster = MarkerCluster(name="Fire Risk Markers").add_to(m)

    # Risk level color mapping
    def get_color(score):
        if score < 25:
            return 'green'
        elif score < 50:
            return 'orange'
        elif score < 75:
            return 'darkorange'
        else:
            return 'red'

    # Plot dataset sample points
    for idx, row in df.iterrows():
        # Compute proxy risk score for dataset rows
        score = float(row.get('fire_weather_index_proxy', 30))
        color = get_color(score)

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; color: #222; width: 220px;">
            <h4 style="margin:0 0 5px 0; color: #111;">Region: <b>{row.get('region', 'N/A')}</b></h4>
            <hr style="margin: 3px 0;">
            <b>Fire Risk Proxy:</b> {score:.1f}%<br>
            <b>Status:</b> {'Fire Event' if row.get('fire_occurred') == 1 else 'No Fire'}<br>
            <b>Temp:</b> {row.get('temperature_c', 'N/A')} °C<br>
            <b>Humidity:</b> {row.get('humidity_pct', 'N/A')} %<br>
            <b>Wind:</b> {row.get('wind_speed_kmh', 'N/A')} km/h<br>
            <b>Rainfall:</b> {row.get('rainfall_mm', 'N/A')} mm<br>
            <b>Dryness Index:</b> {row.get('vegetation_dryness_index', 'N/A')}/100<br>
            <b>Lat/Lon:</b> {row.get('latitude')}, {row.get('longitude')}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row.get('region', 'Area')}: Risk {score:.0f}%",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(marker_cluster)

    # Highlight user predicted point if supplied
    if predicted_point:
        plat = predicted_point['latitude']
        plon = predicted_point['longitude']
        pscore = predicted_point['risk_score']
        pcat = predicted_point['category']
        pcolor = predicted_point['color']

        pred_popup = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; color: #111; width: 230px;">
            <h3 style="margin:0 0 5px 0; color: {pcolor};">📍 User Prediction Target</h3>
            <hr style="margin: 3px 0;">
            <b>Fire Risk Score:</b> <span style="font-size:15px; font-weight:bold;">{pscore}%</span> ({pcat})<br>
            <b>Temperature:</b> {predicted_point.get('temperature_c')} °C<br>
            <b>Humidity:</b> {predicted_point.get('humidity_pct')} %<br>
            <b>Wind Speed:</b> {predicted_point.get('wind_speed_kmh')} km/h<br>
            <b>Vegetation Dryness:</b> {predicted_point.get('vegetation_dryness_index')}/100<br>
            <b>Coordinates:</b> {plat:.4f}, {plon:.4f}
        </div>
        """

        folium.Marker(
            location=[plat, plon],
            popup=folium.Popup(pred_popup, max_width=260),
            tooltip=f"PREDICTED TARGET: {pscore}% ({pcat})",
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #2d3142;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        color: #8b92a7;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-change { font-size: 13px; color: #10b981; }
    .metric-change.negative { color: #ef4444; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e2130;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b92a7;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ef4444 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def get_aqi_category(aqi):
    """Return AQI category and color based on value"""
    if aqi <= 50:
        return "Good", "#10b981"
    elif aqi <= 100:
        return "Moderate", "#f59e0b"
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "#ff6b6b"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "#c026d3"
    else:
        return "Hazardous", "#7c2d12"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\hp\Documents\karachi_air_with_aqi_simple_2024.csv')
        df.columns = df.columns.str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def forecast_simple(df, days=3):
    """Simple forecast based on historical patterns"""
    try:
        df_sorted = df.sort_values('timestamp').copy()
        
        # Get day of week patterns
        df_sorted['dow'] = df_sorted['timestamp'].dt.dayofweek
        daily_aqi = df_sorted.groupby('dow')['aqi'].agg(['mean', 'std']).to_dict('index')
        daily_temp = df_sorted.groupby('dow')['temp'].mean().to_dict()
        
        # Recent trend (last 14 days)
        recent = df_sorted.tail(336)  # ~14 days of hourly data
        recent_aqi = recent['aqi'].mean()
        recent_temp = recent['temp'].mean()
        
        # Calculate simple trend
        recent_aqi_last3 = recent.tail(72)['aqi'].mean()
        aqi_trend = recent_aqi_last3 - recent_aqi
        
        forecasts = []
        
        for day in range(1, days + 1):
            fdate = datetime.now() + timedelta(days=day)
            dow = fdate.weekday()
            
            # Get historical pattern for this day of week
            if dow in daily_aqi:
                base_aqi = daily_aqi[dow]['mean']
                aqi_std = daily_aqi[dow]['std']
            else:
                base_aqi = recent_aqi
                aqi_std = recent['aqi'].std()
            
            base_temp = daily_temp.get(dow, recent_temp)
            
            # Apply trend and add slight variation
            predicted_aqi = base_aqi + (aqi_trend * day * 0.3) + np.random.uniform(-aqi_std*0.2, aqi_std*0.2)
            predicted_temp = base_temp + np.random.uniform(-1, 1)
            
            forecasts.append({
                'day': day,
                'aqi': np.clip(predicted_aqi, 10, 500),
                'temp': np.clip(predicted_temp, 15, 45),
                'date': fdate
            })
        
        return forecasts
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None and len(df) > 0:
    st.title("🌤️ Pearls AQI Dashboard")
    st.markdown("---")
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_temp = df['temp'].iloc[-1]
    avg_temp = df['temp'].mean()
    temp_change = ((current_temp - avg_temp) / avg_temp) * 100
    
    current_humidity = df['humidity'].iloc[-1]
    avg_humidity = df['humidity'].mean()
    humidity_change = current_humidity - avg_humidity
    
    current_wind = df['wind_speed'].iloc[-1]
    max_wind = df['wind_speed'].max()
    
    with col1:
        st.metric("Temperature", f"{current_temp:.1f}°C", f"{temp_change:+.1f}%")
    
    with col2:
        st.metric("Humidity", f"{current_humidity:.1f}%", f"{humidity_change:+.1f}%")
    
    with col3:
        st.metric("Wind Speed", f"{current_wind:.1f} m/s", f"Max: {max_wind:.1f} m/s")
    
    with col4:
        current_aqi = df['aqi'].iloc[-1]
        aqi_cat, aqi_color = get_aqi_category(current_aqi)
        st.metric("Current AQI", f"{current_aqi:.0f}", aqi_cat)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Forecast", "💡 Insights"])
    
    with tab1:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
            monthly_data = df.groupby('year_month').agg({
                'temp': 'mean',
                'humidity': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_data['year_month'], y=monthly_data['temp'],
                mode='lines+markers', name='Temperature',
                line=dict(color='#3b82f6', width=3), marker=dict(size=8),
                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            fig.add_trace(go.Scatter(
                x=monthly_data['year_month'], y=monthly_data['humidity'],
                mode='lines+markers', name='Humidity',
                line=dict(color='#ef4444', width=3), marker=dict(size=8),
                yaxis='y2'
            ))
            fig.update_layout(
                title='Monthly Temperature & Humidity Trends',
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font=dict(color='#ffffff'), height=500,
                xaxis=dict(showgrid=True, gridcolor='#2d3142', title='Month'),
                yaxis=dict(showgrid=True, gridcolor='#2d3142',
                          title=dict(text='Temperature (°C)', font=dict(color='#3b82f6'))),
                yaxis2=dict(title=dict(text='Humidity (%)', font=dict(color='#ef4444')),
                           overlaying='y', side='right'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("<h3 style='color: #ffffff; margin-bottom: 24px;'>📊 Statistics</h3>", unsafe_allow_html=True)
            
            temp_stats = df['temp'].describe()
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e2130 0%, #2d3142 100%); 
                        border-radius: 12px; padding: 20px; margin-bottom: 20px; 
                        border: 1px solid #3b82f6;'>
                <p style='color: #3b82f6; font-weight: bold; font-size: 14px; 
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>
                    🌡️ Temperature
                </p>
                <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>MEAN</p>
                        <p style='color: #ffffff; font-size: 20px; font-weight: bold; margin: 0;'>{:.2f}°C</p>
                    </div>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>STD DEV</p>
                        <p style='color: #ffffff; font-size: 20px; font-weight: bold; margin: 0;'>{:.2f}</p>
                    </div>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>MIN</p>
                        <p style='color: #3b82f6; font-size: 18px; font-weight: bold; margin: 0;'>{:.2f}°C</p>
                    </div>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>MAX</p>
                        <p style='color: #ef4444; font-size: 18px; font-weight: bold; margin: 0;'>{:.2f}°C</p>
                    </div>
                </div>
            </div>
            """.format(temp_stats['mean'], temp_stats['std'], temp_stats['min'], temp_stats['max']), unsafe_allow_html=True)
            
            hum_stats = df['humidity'].describe()
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e2130 0%, #2d3142 100%); 
                        border-radius: 12px; padding: 20px; margin-bottom: 20px; 
                        border: 1px solid #10b981;'>
                <p style='color: #10b981; font-weight: bold; font-size: 14px; 
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>
                    💧 Humidity
                </p>
                <div style='display: flex; justify-content: space-between;'>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>MEAN</p>
                        <p style='color: #ffffff; font-size: 20px; font-weight: bold; margin: 0;'>{:.2f}%</p>
                    </div>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>STD DEV</p>
                        <p style='color: #ffffff; font-size: 20px; font-weight: bold; margin: 0;'>{:.2f}</p>
                    </div>
                </div>
            </div>
            """.format(hum_stats['mean'], hum_stats['std']), unsafe_allow_html=True)
            
            aqi_stats = df['aqi'].describe()
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e2130 0%, #2d3142 100%); 
                        border-radius: 12px; padding: 20px; 
                        border: 1px solid #f59e0b;'>
                <p style='color: #f59e0b; font-weight: bold; font-size: 14px; 
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>
                    🌫️ Air Quality Index
                </p>
                <div style='display: flex; justify-content: space-between;'>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>MEAN</p>
                        <p style='color: #ffffff; font-size: 20px; font-weight: bold; margin: 0;'>{:.0f}</p>
                    </div>
                    <div>
                        <p style='color: #8b92a7; font-size: 11px; margin: 0;'>MAX</p>
                        <p style='color: #f59e0b; font-size: 20px; font-weight: bold; margin: 0;'>{:.0f}</p>
                    </div>
                </div>
            </div>
            """.format(aqi_stats['mean'], aqi_stats['max']), unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3 style='color: #ffffff;'>🔮 3-Day AQI Forecast</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #8b92a7;'>*Predictions based on historical patterns*</p>", unsafe_allow_html=True)
        
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Analyzing patterns..."):
                forecasts = forecast_simple(df, days=3)
                
                if forecasts:
                    st.markdown("<hr style='border-color: #2d3142;'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #ffffff;'>📅 Daily Forecasts</h4>", unsafe_allow_html=True)
                    fc1, fc2, fc3 = st.columns(3)
                    
                    for i, fc in enumerate(forecasts):
                        with [fc1, fc2, fc3][i]:
                            pred_aqi = fc['aqi']
                            pred_temp = fc['temp']
                            fdate = fc['date']
                            
                            aqi_cat, aqi_color = get_aqi_category(pred_aqi)
                            
                            if pred_temp > 35:
                                temp_emoji = "🔥"
                            elif pred_temp > 30:
                                temp_emoji = "☀️"
                            elif pred_temp > 25:
                                temp_emoji = "🌤️"
                            else:
                                temp_emoji = "😊"
                            
                            with st.container():
                                st.markdown(f"<p style='color: #ffffff; font-weight: bold; font-size: 16px;'>DAY {fc['day']} - {fdate.strftime('%a, %b %d')}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: #8b92a7; font-size: 12px; font-weight: 600;'>AIR QUALITY INDEX</p>", unsafe_allow_html=True)
                                st.markdown(f"<h2 style='text-align: center; font-size: 48px; color: #ffffff; margin: 10px 0;'>{pred_aqi:.0f}</h2>", unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align: center;'><span style='background-color: {aqi_color}; color: white; padding: 8px 20px; border-radius: 20px; font-size: 14px; font-weight: 600;'>{aqi_cat}</span></div>", unsafe_allow_html=True)
                                st.markdown("<hr style='margin: 20px 0; border-color: #2d3142;'>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: #8b92a7; font-size: 11px; font-weight: 600;'>TEMPERATURE</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; font-size: 32px; margin: 5px 0;'>{temp_emoji}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; font-size: 24px; font-weight: 600; color: #ffffff;'>{pred_temp:.1f}°C</p>", unsafe_allow_html=True)
                    
                    st.success("✅ Forecast generated based on historical patterns!")
    
    with tab3:
        st.markdown("<h3 style='color: #ffffff;'>💡 Karachi Weather Insights</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            recent_t = df['temp'].tail(24).mean()
            recent_h = df['humidity'].tail(24).mean()
            heat_idx = recent_t + (0.5 * (recent_h - 50) / 10)
            
            if heat_idx > 35:
                adv = "Stay indoors • High heat risk"
            elif heat_idx > 30:
                adv = "Early morning walks recommended"
            else:
                adv = "Perfect for outdoor activities"
            
            st.markdown(f"<p style='color: #ffffff; font-weight: bold; font-size: 16px;'>🔥 Heat Index</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #ffffff; font-size: 28px; font-weight: bold;'>{heat_idx:.1f}°C</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #8b92a7; font-size: 14px;'>{adv}</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            try:
                dft = df.copy()
                dft['hour'] = dft['timestamp'].dt.hour
                hw = dft.groupby('hour')['wind_speed'].mean()
                bw = hw.idxmax()
                st.markdown(f"<p style='color: #ffffff; font-weight: bold; font-size: 16px;'>🌊 Best Sea Breeze</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ffffff; font-size: 28px; font-weight: bold;'>{int(bw)}:00</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #8b92a7; font-size: 14px;'>Perfect for beach walk</p>", unsafe_allow_html=True)
            except:
                pass
        
        with c2:
            try:
                dft = df.copy()
                dft['hour'] = dft['timestamp'].dt.hour
                ht = dft.groupby('hour')['temp'].mean()
                ph = ht.idxmax()
                pt = ht.max()
                st.markdown(f"<p style='color: #ffffff; font-weight: bold; font-size: 16px;'>⚡ Peak Heat (Avoid)</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ffffff; font-size: 28px; font-weight: bold;'>{int(ph)}:00</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #8b92a7; font-size: 14px;'>{pt:.1f}°C • Stay indoors</p>", unsafe_allow_html=True)
            except:
                pass
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            try:
                dft = df.copy()
                dft['hour'] = dft['timestamp'].dt.hour
                ct = dft.groupby('hour')['temp'].mean()
                ch = ct.idxmin()
                ctemp = ct.min()
                st.markdown(f"<p style='color: #ffffff; font-weight: bold; font-size: 16px;'>💧 Best Walk Time</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #ffffff; font-size: 28px; font-weight: bold;'>{int(ch)}:00</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #8b92a7; font-size: 14px;'>{ctemp:.1f}°C • Ideal for jogging</p>", unsafe_allow_html=True)
            except:
                pass

else:
    st.error("❌ Could not load data")
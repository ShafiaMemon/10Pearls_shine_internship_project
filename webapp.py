import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")

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

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\hp\Documents\karachi_air_weather_2024.csv')
        df.columns = df.columns.str.strip()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def forecast_weather(df, days=3):
    try:
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['hour'] = df_sorted['timestamp'].dt.hour
        
        hourly_temp = df_sorted.groupby('hour')['temp_C'].mean()
        hourly_humidity = df_sorted.groupby('hour')['humidity_percent'].mean()
        hourly_wind = df_sorted.groupby('hour')['wind_speed_mps'].mean()
        
        last_timestamp = df_sorted['timestamp'].max()
        recent = df_sorted[df_sorted['timestamp'] >= (last_timestamp - timedelta(days=7))]
        
        def calc_trend(data, col):
            data = data.copy()
            data['hrs'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600
            x = data['hrs'].values
            y = data[col].values
            mask = ~np.isnan(y)
            x, y = x[mask], y[mask]
            if len(x) < 2:
                return 0
            slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
            return slope
        
        temp_slope = calc_trend(recent, 'temp_C')
        hum_slope = calc_trend(recent, 'humidity_percent')
        wind_slope = calc_trend(recent, 'wind_speed_mps')
        
        base_temp = recent['temp_C'].tail(24).mean()
        base_hum = recent['humidity_percent'].tail(24).mean()
        base_wind = recent['wind_speed_mps'].tail(24).mean()
        
        df_sorted['dow'] = df_sorted['timestamp'].dt.dayofweek
        weekly_temp = df_sorted.groupby('dow')['temp_C'].mean()
        weekly_hum = df_sorted.groupby('dow')['humidity_percent'].mean()
        
        forecasts = []
        last_hrs = (recent['timestamp'].max() - recent['timestamp'].min()).total_seconds() / 3600
        
        for day in range(1, days + 1):
            fc = {'day': day, 'temps': [], 'humidity': [], 'wind': []}
            fdate = datetime.now() + timedelta(days=day)
            dow = fdate.weekday()
            
            weekly_t = weekly_temp.get(dow, base_temp) - weekly_temp.mean()
            weekly_h = weekly_hum.get(dow, base_hum) - weekly_hum.mean()
            
            for hr in range(24):
                hr_off = last_hrs + (day - 1) * 24 + hr
                t_seas = hourly_temp.get(hr, base_temp) - hourly_temp.mean()
                h_seas = hourly_humidity.get(hr, base_hum) - hourly_humidity.mean()
                w_seas = hourly_wind.get(hr, base_wind) - hourly_wind.mean()
                
                pred_t = base_temp + t_seas + weekly_t + temp_slope * hr_off + np.random.normal(0, 0.3)
                pred_h = base_hum + h_seas + weekly_h + hum_slope * hr_off + np.random.normal(0, 1.5)
                pred_w = base_wind + w_seas + wind_slope * hr_off + np.random.normal(0, 0.2)
                
                fc['temps'].append(np.clip(pred_t, 15, 45))
                fc['humidity'].append(np.clip(pred_h, 20, 100))
                fc['wind'].append(np.clip(pred_w, 0, 25))
            
            forecasts.append(fc)
        
        return forecasts, {'temp_trend': temp_slope, 'humidity_trend': hum_slope, 'wind_trend': wind_slope}
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return None, None

df = load_data()

if df is not None and len(df) > 0:
    st.title("🌤️ Pearls AQI Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_temp = df['temp_C'].iloc[-1]
    avg_temp = df['temp_C'].mean()
    temp_change = ((current_temp - avg_temp) / avg_temp) * 100
    
    current_humidity = df['humidity_percent'].iloc[-1]
    avg_humidity = df['humidity_percent'].mean()
    humidity_change = current_humidity - avg_humidity
    
    current_wind = df['wind_speed_mps'].iloc[-1]
    max_wind = df['wind_speed_mps'].max()
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Temperature</div>
            <div class='metric-value'>{current_temp:.1f}°C</div>
            <div class='metric-change {"negative" if temp_change < 0 else ""}'>
                {"↓" if temp_change < 0 else "↑"} {abs(temp_change):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Humidity</div>
            <div class='metric-value'>{current_humidity:.1f}%</div>
            <div class='metric-change {"negative" if humidity_change < 0 else ""}'>
                {"↓" if humidity_change < 0 else "↑"} {abs(humidity_change):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Wind Speed</div>
            <div class='metric-value'>{current_wind:.1f} m/s</div>
            <div class='metric-change'>Max: {max_wind:.1f} m/s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Records</div>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-change'>Data points</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Forecast", "💡 Insights"])
    
    with tab1:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
            monthly_data = df.groupby('year_month').agg({
                'temp_C': 'mean',
                'humidity_percent': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_data['year_month'], y=monthly_data['temp_C'],
                mode='lines+markers', name='Temperature',
                line=dict(color='#3b82f6', width=3), marker=dict(size=8),
                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            fig.add_trace(go.Scatter(
                x=monthly_data['year_month'], y=monthly_data['humidity_percent'],
                mode='lines+markers', name='Humidity',
                line=dict(color='#ef4444', width=3), marker=dict(size=8),
                yaxis='y2'
            ))
            fig.update_layout(
                title='Monthly Temperature & Humidity Trends',
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font=dict(color='#ffffff'), height=400,
                xaxis=dict(showgrid=True, gridcolor='#2d3142', title='Month'),
                yaxis=dict(showgrid=True, gridcolor='#2d3142',
                          title=dict(text='Temperature (°C)', font=dict(color='#3b82f6'))),
                yaxis2=dict(title=dict(text='Humidity (%)', font=dict(color='#ef4444')),
                           overlaying='y', side='right'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🕒 PM2.5 Air Quality Trend")
            try:
                pm_data = df.sort_values('timestamp')
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=pm_data['timestamp'], y=pm_data['pm2_5_ugm3'],
                    mode='lines', name='PM2.5',
                    line=dict(color='#14b8a6', width=2),
                    fill='tozeroy', fillcolor='rgba(20, 184, 166, 0.1)'
                ))
                fig2.add_hline(y=35, line_dash="dash", line_color="#f59e0b", 
                              annotation_text="Moderate", annotation_position="right")
                fig2.add_hline(y=55, line_dash="dash", line_color="#ef4444",
                              annotation_text="Unhealthy", annotation_position="right")
                fig2.update_layout(
                    plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                    font=dict(color='#ffffff'), height=400,
                    xaxis=dict(showgrid=True, gridcolor='#2d3142', title='Timestamp'),
                    yaxis=dict(showgrid=True, gridcolor='#2d3142', title='PM2.5 (µg/m³)'),
                    showlegend=False, hovermode='x unified'
                )
                st.plotly_chart(fig2, use_container_width=True)
            except:
                st.warning("PM2.5 data not available")
        
        with col_right:
            st.markdown("### 📊 Statistics")
            temp_stats = df['temp_C'].describe()
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom: 16px;'>
                <div class='metric-label'>🌡️ Temperature</div>
                <div style='display: flex; justify-content: space-between; margin-top: 12px;'>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>MEAN</div>
                        <div style='color: #fff; font-size: 18px; font-weight: 600;'>{temp_stats['mean']:.2f}°C</div>
                    </div>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>STD</div>
                        <div style='color: #fff; font-size: 18px; font-weight: 600;'>{temp_stats['std']:.2f}</div>
                    </div>
                </div>
                <div style='display: flex; justify-content: space-between; margin-top: 12px;'>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>MIN</div>
                        <div style='color: #3b82f6; font-size: 16px; font-weight: 600;'>{temp_stats['min']:.2f}°C</div>
                    </div>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>MAX</div>
                        <div style='color: #ef4444; font-size: 16px; font-weight: 600;'>{temp_stats['max']:.2f}°C</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            hum_stats = df['humidity_percent'].describe()
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom: 16px;'>
                <div class='metric-label'>💧 Humidity</div>
                <div style='display: flex; justify-content: space-between; margin-top: 12px;'>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>MEAN</div>
                        <div style='color: #fff; font-size: 18px; font-weight: 600;'>{hum_stats['mean']:.2f}%</div>
                    </div>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>STD</div>
                        <div style='color: #fff; font-size: 18px; font-weight: 600;'>{hum_stats['std']:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            wind_stats = df['wind_speed_mps'].describe()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🌬️ Wind Speed</div>
                <div style='display: flex; justify-content: space-between; margin-top: 12px;'>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>MEAN</div>
                        <div style='color: #fff; font-size: 18px; font-weight: 600;'>{wind_stats['mean']:.2f} m/s</div>
                    </div>
                    <div style='flex: 1;'>
                        <div style='color: #8b92a7; font-size: 11px;'>MAX</div>
                        <div style='color: #f59e0b; font-size: 18px; font-weight: 600;'>{wind_stats['max']:.2f} m/s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🔮 3-Day Weather Forecast")
        st.markdown("*Based on historical patterns from your dataset*")
        
        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Analyzing patterns..."):
                forecasts, trends = forecast_weather(df, days=3)
                
                if forecasts:
                    st.markdown("#### 📊 Detected Trends")
                    tc1, tc2, tc3 = st.columns(3)
                    
                    with tc1:
                        td = "📈 Rising" if trends['temp_trend'] > 0 else "📉 Falling"
                        st.info(f"**Temp**: {td} ({trends['temp_trend']:.3f}°C/hr)")
                    with tc2:
                        hd = "📈 Rising" if trends['humidity_trend'] > 0 else "📉 Falling"
                        st.info(f"**Humidity**: {hd} ({trends['humidity_trend']:.3f}%/hr)")
                    with tc3:
                        wd = "📈 Rising" if trends['wind_trend'] > 0 else "📉 Falling"
                        st.info(f"**Wind**: {wd} ({trends['wind_trend']:.3f} m/s/hr)")
                    
                    st.markdown("---")
                    fc1, fc2, fc3 = st.columns(3)
                    
                    for i, fc in enumerate(forecasts):
                        with [fc1, fc2, fc3][i]:
                            avg_t = np.mean(fc['temps'])
                            min_t = np.min(fc['temps'])
                            max_t = np.max(fc['temps'])
                            avg_h = np.mean(fc['humidity'])
                            avg_w = np.mean(fc['wind'])
                            
                            if avg_t > 35:
                                cond, color = "🔥 Very Hot", "#ef4444"
                            elif avg_t > 30:
                                cond, color = "☀️ Hot", "#f59e0b"
                            elif avg_t > 25:
                                cond, color = "🌤️ Warm", "#10b981"
                            else:
                                cond, color = "😊 Pleasant", "#3b82f6"
                            
                            st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-label'>Day {fc['day']} - {(datetime.now() + timedelta(days=fc['day'])).strftime('%a, %b %d')}</div>
                                <div style='text-align: center; margin: 16px 0;'>
                                    <div style='font-size: 48px;'>{cond.split()[0]}</div>
                                    <div style='color: {color}; font-size: 16px; font-weight: 600; margin-top: 8px;'>{' '.join(cond.split()[1:])}</div>
                                </div>
                                <div class='metric-value' style='font-size: 36px; text-align: center;'>{avg_t:.1f}°C</div>
                                <div style='text-align: center; color: #8b92a7; font-size: 13px; margin-top: 8px;'>
                                    Range: {min_t:.1f}° - {max_t:.1f}°C
                                </div>
                                <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid #2d3142;'>
                                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                                        <span style='color: #8b92a7; font-size: 12px;'>💧 Humidity</span>
                                        <span style='color: #fff; font-weight: 600;'>{avg_h:.0f}%</span>
                                    </div>
                                    <div style='display: flex; justify-content: space-between;'>
                                        <span style='color: #8b92a7; font-size: 12px;'>🌬️ Wind</span>
                                        <span style='color: #fff; font-weight: 600;'>{avg_w:.1f} m/s</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("#### 📈 Hourly Temperature Forecast")
                    
                    fig_fc = go.Figure()
                    for i, fc in enumerate(forecasts):
                        hrs = [(datetime.now() + timedelta(days=fc['day'], hours=h)) for h in range(24)]
                        fig_fc.add_trace(go.Scatter(
                            x=hrs, y=fc['temps'],
                            mode='lines+markers', name=f'Day {fc["day"]}',
                            line=dict(width=3), marker=dict(size=6)
                        ))
                    
                    fig_fc.update_layout(
                        plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                        font=dict(color='#ffffff'), height=400,
                        xaxis=dict(showgrid=True, gridcolor='#2d3142', title='Date & Time'),
                        yaxis=dict(showgrid=True, gridcolor='#2d3142', title='Temperature (°C)'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success("✅ Forecast complete!")
    
    with tab3:
        st.markdown("### 💡 Karachi Weather Insights")
        c1, c2 = st.columns(2)
        
        with c1:
            recent_t = df['temp_C'].tail(24).mean()
            recent_h = df['humidity_percent'].tail(24).mean()
            heat_idx = recent_t + (0.5 * (recent_h - 50) / 10)
            
            if heat_idx > 35:
                adv = "Stay indoors • High heat risk"
            elif heat_idx > 30:
                adv = "Early morning walks recommended"
            else:
                adv = "Perfect for outdoor activities"
            
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🔥 Heat Index</div>
                <div class='metric-value' style='font-size: 28px;'>{heat_idx:.1f}°C</div>
                <div style='color: #8b92a7; font-size: 13px; margin-top: 8px;'>{adv}</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                dft = df.copy()
                dft['hour'] = dft['timestamp'].dt.hour
                hw = dft.groupby('hour')['wind_speed_mps'].mean()
                bw = hw.idxmax()
                
                st.markdown(f"""
                <div class='metric-card' style='margin-top: 16px;'>
                    <div class='metric-label'>🌊 Best Sea Breeze</div>
                    <div class='metric-value' style='font-size: 28px;'>{int(bw)}:00</div>
                    <div style='color: #8b92a7; font-size: 13px; margin-top: 8px;'>Perfect for beach walk</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass
        
        with c2:
            try:
                dft = df.copy()
                dft['hour'] = dft['timestamp'].dt.hour
                ht = dft.groupby('hour')['temp_C'].mean()
                ph = ht.idxmax()
                pt = ht.max()
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>⚡ Peak Heat (Avoid)</div>
                    <div class='metric-value' style='font-size: 28px;'>{int(ph)}:00</div>
                    <div style='color: #8b92a7; font-size: 13px; margin-top: 8px;'>{pt:.1f}°C • Stay indoors</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass
            
            try:
                dft = df.copy()
                dft['hour'] = dft['timestamp'].dt.hour
                ct = dft.groupby('hour')['temp_C'].mean()
                ch = ct.idxmin()
                ctemp = ct.min()
                
                st.markdown(f"""
                <div class='metric-card' style='margin-top: 16px;'>
                    <div class='metric-label'>💧 Best Walk Time</div>
                    <div class='metric-value' style='font-size: 28px;'>{int(ch)}:00</div>
                    <div style='color: #8b92a7; font-size: 13px; margin-top: 8px;'>{ctemp:.1f}°C • Ideal for jogging</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass

else:
    st.error("❌ Could not load data")
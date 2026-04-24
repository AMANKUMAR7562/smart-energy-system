import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Energy System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# PREMIUM CSS THEME
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 40%, #0a0f1e 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1225 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(16,185,129,0.10) 50%, rgba(59,130,246,0.12) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #34d399 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(30,32,55,0.9) 0%, rgba(20,22,40,0.95) 100%);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.45);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.12);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card.purple::after { background: linear-gradient(90deg, #818cf8, #6366f1); }
.metric-card.green::after  { background: linear-gradient(90deg, #34d399, #10b981); }
.metric-card.blue::after   { background: linear-gradient(90deg, #60a5fa, #3b82f6); }
.metric-card.amber::after  { background: linear-gradient(90deg, #fbbf24, #f59e0b); }

.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.2rem;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

/* ── Section Cards ── */
.glass-card {
    background: linear-gradient(135deg, rgba(30,32,55,0.7) 0%, rgba(15,17,35,0.8) 100%);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title .icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.icon-purple { background: rgba(99,102,241,0.2); }
.icon-green  { background: rgba(16,185,129,0.2); }
.icon-blue   { background: rgba(59,130,246,0.2); }
.icon-amber  { background: rgba(245,158,11,0.2); }

/* ── Insight Cards ── */
.insight-card {
    background: rgba(30,32,55,0.6);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    transition: border-color 0.3s;
}
.insight-card:hover { border-color: rgba(99,102,241,0.3); }
.insight-emoji { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.insight-text { color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; }
.insight-text strong { color: #f1f5f9; }

/* ── Prediction Result ── */
.prediction-result {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(99,102,241,0.10) 100%);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.prediction-value {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #34d399, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}
.pred-warning {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #fca5a5;
    margin-top: 1rem;
    font-size: 0.9rem;
}
.pred-success {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #6ee7b7;
    margin-top: 1rem;
    font-size: 0.9rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(99,102,241,0.08);
    margin-top: 3rem;
}

/* ── Streamlit Overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    box-shadow: 0 8px 30px rgba(99,102,241,0.35) !important;
    transform: translateY(-2px);
}
div[data-testid="stMetric"] { display: none; }

.stSelectbox label, .stSlider label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("smart_home_energy_consumption_large.csv")
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Hour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

    # Build name → code mappings BEFORE encoding (so dropdowns can show names)
    appliance_cat = df['Appliance Type'].astype('category')
    appliance_map = dict(zip(appliance_cat.cat.categories, range(len(appliance_cat.cat.categories))))

    season_cat = df['Season'].astype('category')
    season_map = dict(zip(season_cat.cat.categories, range(len(season_cat.cat.categories))))

    # Now encode the columns for the model
    df['Appliance Type'] = appliance_cat.cat.codes
    df['Season'] = season_cat.cat.codes

    return df, appliance_map, season_map

df, appliance_map, season_map = load_data()

# =========================
# MODEL
# =========================
features = [
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',
    'Outdoor Temperature (°C)', 'Appliance Type',
    'Season', 'Household Size'
]
X = df[features]
y = df['Energy Consumption (kWh)']

@st.cache_resource
def train_model(X, y):
    model = XGBRegressor()
    model.fit(X, y)
    return model

model = train_model(X, y)

# Plotly theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='rgba(99,102,241,0.08)', zerolinecolor='rgba(99,102,241,0.08)'),
    yaxis=dict(gridcolor='rgba(99,102,241,0.08)', zerolinecolor='rgba(99,102,241,0.08)'),
    hoverlabel=dict(bgcolor='#1e1e37', font_color='#e2e8f0', bordercolor='rgba(99,102,241,0.3)'),
)

# =========================
# HERO BANNER
# =========================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⚡ Smart Energy System</div>
    <div class="hero-sub">AI-powered energy prediction & real-time consumption analytics for smart homes</div>
</div>
""", unsafe_allow_html=True)

# =========================
# TOP METRICS
# =========================
total_energy = df['Energy Consumption (kWh)'].sum()
avg_energy = df['Energy Consumption (kWh)'].mean()
max_energy = df['Energy Consumption (kWh)'].max()
total_homes = df['Home ID'].nunique() if 'Home ID' in df.columns else len(df)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card purple">
        <div class="metric-icon">⚡</div>
        <div class="metric-value">{total_energy:,.0f}</div>
        <div class="metric-label">Total kWh</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-icon">📊</div>
        <div class="metric-value">{avg_energy:.2f}</div>
        <div class="metric-label">Avg Consumption</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card blue">
        <div class="metric-icon">🔺</div>
        <div class="metric-value">{max_energy:.2f}</div>
        <div class="metric-label">Peak Usage (kWh)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card amber">
        <div class="metric-icon">🏠</div>
        <div class="metric-value">{total_homes}</div>
        <div class="metric-label">Homes Monitored</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# SIDEBAR — PREDICTION INPUTS
# =========================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div style="font-size:2.5rem;">⚡</div>
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:0.3rem;">Energy Predictor</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.2rem;">XGBoost ML Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⏱️ Time Settings")
    hour = st.slider("Hour of Day", 0, 23, 12)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day of Week (0=Mon)", 0, 6, 3)
    weekend = st.selectbox("Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("---")
    st.markdown("##### 🏠 Home Settings")
    temperature = st.slider("Temperature (°C)", -10.0, 45.0, 25.0)
    household = st.slider("Household Size", 1, 6, 3)
    appliance_name = st.selectbox("Appliance Type", sorted(appliance_map.keys()))
    season_name = st.selectbox("Season", sorted(season_map.keys()))

    # Look up the encoded numerical value for the model
    appliance = appliance_map[appliance_name]
    season = season_map[season_name]

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Energy Usage")

# =========================
# MAIN CONTENT — TWO COLUMNS
# =========================
col_left, col_right = st.columns([3, 2])

# ── Reverse mapping: code → name (for chart labels) ──
code_to_appliance = {v: k for k, v in appliance_map.items()}

# ── LEFT: CHARTS ──
with col_left:
    # 1) Daily Average Energy Trend (clean, aggregated by date)
    show_trend = st.checkbox("📈 Daily Energy Consumption Trend", value=False)
    if show_trend:
        daily = df.groupby(df['DateTime'].dt.date)['Energy Consumption (kWh)'].mean().reset_index()
        daily.columns = ['Date', 'Avg kWh']
        daily = daily.sort_values('Date')
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily['Date'], y=daily['Avg kWh'],
            mode='lines', fill='tozeroy',
            line=dict(color='#818cf8', width=2, shape='spline'),
            fillcolor='rgba(99,102,241,0.08)',
            hovertemplate='%{x}<br>Avg: %{y:.2f} kWh<extra></extra>'
        ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT, height=280, title=None,
            xaxis_title='Date', yaxis_title='Avg kWh',
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Two charts side by side
    ch1, ch2 = st.columns(2)
    with ch1:
        # 2) Average Consumption by Hour
        show_hourly = st.checkbox("⏰ Hourly Pattern", value=False)
        if show_hourly:
            hourly = df.groupby('Hour')['Energy Consumption (kWh)'].mean().reset_index()
            hourly['Label'] = hourly['Hour'].apply(
                lambda h: f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
            )
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=hourly['Label'], y=hourly['Energy Consumption (kWh)'],
                marker=dict(
                    color=hourly['Energy Consumption (kWh)'],
                    colorscale=[[0, '#6366f1'], [0.5, '#818cf8'], [1, '#34d399']],
                    cornerradius=4
                ),
                hovertemplate='%{x}<br>%{y:.2f} kWh<extra></extra>'
            ))
            fig_hourly.update_layout(
                **PLOTLY_LAYOUT, height=280, showlegend=False,
                xaxis_title='Hour', yaxis_title='Avg kWh',
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

    with ch2:
        # 3) Avg Consumption by Temperature Range (binned)
        show_temp = st.checkbox("🌡️ Temp vs Usage", value=False)
        if show_temp:
            temp_df = df.copy()
            temp_df['Temp Bin'] = pd.cut(
                temp_df['Outdoor Temperature (°C)'],
                bins=range(-15, 50, 5),
                right=False
            )
            temp_agg = temp_df.groupby('Temp Bin', observed=True)['Energy Consumption (kWh)'].mean().reset_index()
            temp_agg['Temp Label'] = temp_agg['Temp Bin'].astype(str)
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Bar(
                x=temp_agg['Temp Label'], y=temp_agg['Energy Consumption (kWh)'],
                marker=dict(
                    color=temp_agg['Energy Consumption (kWh)'],
                    colorscale=[[0, '#6366f1'], [0.5, '#34d399'], [1, '#f59e0b']],
                    cornerradius=4
                ),
                hovertemplate='%{x} °C<br>Avg: %{y:.2f} kWh<extra></extra>'
            ))
            fig_temp.update_layout(
                **PLOTLY_LAYOUT, height=280, showlegend=False,
                xaxis_title='Temperature Range (°C)', yaxis_title='Avg kWh',
            )
            st.plotly_chart(fig_temp, use_container_width=True)

    # 4) Average Consumption by Appliance (with real names)
    show_appliance = st.checkbox("🔌 Consumption by Appliance Type", value=False)
    if show_appliance:
        app_data = df.groupby('Appliance Type')['Energy Consumption (kWh)'].mean().reset_index()
        app_data['Appliance Name'] = app_data['Appliance Type'].map(code_to_appliance)
        app_data = app_data.sort_values('Energy Consumption (kWh)', ascending=True)
        fig_app = go.Figure()
        fig_app.add_trace(go.Bar(
            x=app_data['Energy Consumption (kWh)'], y=app_data['Appliance Name'],
            orientation='h',
            marker=dict(
                color=app_data['Energy Consumption (kWh)'],
                colorscale=[[0, '#6366f1'], [0.5, '#818cf8'], [1, '#34d399']],
                cornerradius=4
            ),
            hovertemplate='%{y}<br>Avg: %{x:.2f} kWh<extra></extra>'
        ))
        fig_app.update_layout(
            **PLOTLY_LAYOUT, height=300, showlegend=False,
            xaxis_title='Avg kWh', yaxis_title='',
        )
        st.plotly_chart(fig_app, use_container_width=True)

# ── RIGHT: PREDICTION + INSIGHTS ──
with col_right:
    st.markdown("""<div class="section-title">
        <span class="icon icon-purple">🤖</span> Prediction Result
    </div>""", unsafe_allow_html=True)

    if predict_btn:
        input_data = np.array([[hour, month, day, weekend,
                                temperature, appliance, season, household]])
        prediction = model.predict(input_data)[0]
        pct = min(prediction / max_energy * 100, 100)

        st.markdown(f"""
        <div class="prediction-result">
            <div class="prediction-value">{prediction:.2f}</div>
            <div class="prediction-label">Predicted kWh</div>
        </div>
        """, unsafe_allow_html=True)

        if prediction > avg_energy:
            st.markdown(f"""<div class="pred-warning">
                ⚠️ High consumption predicted — <strong>{pct:.0f}%</strong> of peak. Consider reducing usage.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="pred-success">
                ✅ Efficient usage — only <strong>{pct:.0f}%</strong> of peak capacity.
            </div>""", unsafe_allow_html=True)

        # Mini gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number=dict(suffix=" kWh", font=dict(size=24, color='#e2e8f0')),
            gauge=dict(
                axis=dict(range=[0, max_energy], tickcolor='#475569'),
                bar=dict(color='#818cf8'),
                bgcolor='rgba(30,32,55,0.5)',
                bordercolor='rgba(99,102,241,0.2)',
                steps=[
                    dict(range=[0, avg_energy], color='rgba(16,185,129,0.15)'),
                    dict(range=[avg_energy, max_energy], color='rgba(239,68,68,0.10)'),
                ],
                threshold=dict(line=dict(color='#f59e0b', width=2), value=avg_energy)
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#94a3b8'), height=220,
            margin=dict(l=30, r=30, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.markdown("""
        <div class="prediction-result" style="padding:3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🎯</div>
            <div style="color:#94a3b8; font-size:0.95rem;">
                Adjust parameters in the sidebar and click<br>
                <strong style="color:#818cf8;">Predict Energy Usage</strong> to get results
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Smart Insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-title">
        <span class="icon icon-green">🧠</span> Smart Insights
    </div>""", unsafe_allow_html=True)

    peak_hour = df.groupby('Hour')['Energy Consumption (kWh)'].mean().idxmax()
    low_hour = df.groupby('Hour')['Energy Consumption (kWh)'].mean().idxmin()
    weekend_avg = df[df['IsWeekend']==1]['Energy Consumption (kWh)'].mean()
    weekday_avg = df[df['IsWeekend']==0]['Energy Consumption (kWh)'].mean()
    we_diff = ((weekend_avg - weekday_avg) / weekday_avg * 100)

    insights = [
        ("🔥", f"Peak usage at <strong>hour {peak_hour}:00</strong> — schedule heavy loads before or after."),
        ("🌙", f"Lowest consumption at <strong>hour {low_hour}:00</strong> — ideal for automated tasks."),
        ("📅", f"Weekend usage is <strong>{abs(we_diff):.1f}% {'higher' if we_diff > 0 else 'lower'}</strong> than weekdays."),
        ("💡", "Maintain <strong>optimal thermostat</strong> settings to cut heating/cooling waste."),
        ("🔌", "Turn off idle appliances — standby power adds up to <strong>10% of total</strong> usage."),
    ]
    for emoji, text in insights:
        st.markdown(f"""<div class="insight-card">
            <span class="insight-emoji">{emoji}</span>
            <span class="insight-text">{text}</span>
        </div>""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Smart Energy System v2.0 · Powered by XGBoost · Built with Streamlit & Plotly<br>
    © 2026 Smart Energy Analytics — All rights reserved
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Energy System",
    page_icon="⚡",
    layout="wide"
)

# =========================
# CUSTOM STYLE (🔥 UI BOOST)
# =========================
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
}
.metric-box {
    background-color:#1e1e2f;
    padding:15px;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">⚡ Smart Energy Prediction & Analysis System</p>', unsafe_allow_html=True)

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

    df['Appliance Type'] = df['Appliance Type'].astype('category').cat.codes
    df['Season'] = df['Season'].astype('category').cat.codes

    return df

df = load_data()

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

model = XGBRegressor()
model.fit(X, y)

# =========================
# TOP METRICS (🔥 IMPRESSIVE)
# =========================
colA, colB, colC = st.columns(3)

with colA:
    st.metric("Total Energy", f"{df['Energy Consumption (kWh)'].sum():.0f} kWh")

with colB:
    st.metric("Avg Consumption", f"{df['Energy Consumption (kWh)'].mean():.2f} kWh")

with colC:
    st.metric("Max Usage", f"{df['Energy Consumption (kWh)'].max():.2f} kWh")

st.markdown("---")

# =========================
# INPUT + OUTPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔢 Enter Parameters")

    hour = st.slider("Hour of Day", 0, 23, 12)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day of Week", 0, 6, 3)
    weekend = st.selectbox("Weekend?", [0,1])

    temperature = st.slider("Temperature (°C)", -10.0, 45.0, 25.0)
    household = st.slider("Household Size", 1, 6, 3)

    appliance = st.selectbox("Appliance Type", df['Appliance Type'].unique())
    season = st.selectbox("Season", df['Season'].unique())

with col2:
    st.subheader("⚡ Prediction")

    if st.button("🚀 Predict Energy"):

        input_data = np.array([[hour, month, day, weekend,
                                temperature, appliance, season, household]])

        prediction = model.predict(input_data)[0]

        st.success(f"⚡ {prediction:.2f} kWh predicted")

        # Progress bar (🔥 visual boost)
        progress = min(prediction / df['Energy Consumption (kWh)'].max(), 1.0)
        st.progress(progress)

        if prediction > df['Energy Consumption (kWh)'].mean():
            st.error("⚠️ High consumption expected! Reduce usage.")
        else:
            st.success("✅ Efficient energy usage!")

st.markdown("---")

# =========================
# DASHBOARD
# =========================
st.header("📊 Energy Dashboard")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Energy Trend")
    st.line_chart(df['Energy Consumption (kWh)'][:300])

with col4:
    st.subheader("🔥 Appliance Consumption")
    top_appliances = df.groupby('Appliance Type')['Energy Consumption (kWh)'].sum()
    st.bar_chart(top_appliances)

col5, col6 = st.columns(2)

with col5:
    st.subheader("⏰ Hourly Pattern")
    hourly = df.groupby('Hour')['Energy Consumption (kWh)'].mean()
    st.line_chart(hourly)

with col6:
    st.subheader("🌡️ Temperature Impact")
    st.scatter_chart(df[['Outdoor Temperature (°C)', 'Energy Consumption (kWh)']])

st.markdown("---")

# =========================
# SMART INSIGHTS
# =========================
st.header("🧠 Smart Insights")

peak_hour = df.groupby('Hour')['Energy Consumption (kWh)'].mean().idxmax()

st.write(f"🔥 Peak energy usage occurs at hour: **{peak_hour}**")
st.write("💡 Recommendation: Shift heavy appliance usage to off-peak hours")
st.write("🌡️ Maintain optimal temperature settings to reduce energy waste")
st.write("🔌 Turn off unused appliances to save electricity")

st.success("✅ System Ready | High-Performance Smart Energy Model")
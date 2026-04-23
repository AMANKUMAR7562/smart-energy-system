import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Smart Energy System", layout="wide")

st.title("⚡ Smart Energy Prediction & Analysis System")

# =========================
# LOAD DATA (FAST)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("smart_home_energy_consumption_large.csv")

    # Reduce size (IMPORTANT for speed)
    df = df.sample(3000)

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    df['Hour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)

    # Convert categories
    df['Appliance Type'] = df['Appliance Type'].astype('category').cat.codes
    df['Season'] = df['Season'].astype('category').cat.codes

    return df

df = load_data()

# =========================
# MODEL (CACHED)
# =========================
@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(n_estimators=50, max_depth=3)
    model.fit(X, y)
    return model

features = [
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',
    'Outdoor Temperature (°C)', 'Appliance Type',
    'Season', 'Household Size'
]

X = df[features]
y = df['Energy Consumption (kWh)']

model = train_model(X, y)

# =========================
# TOP METRICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Energy", f"{df['Energy Consumption (kWh)'].sum():.0f}")
col2.metric("Average Usage", f"{df['Energy Consumption (kWh)'].mean():.2f}")
col3.metric("Max Usage", f"{df['Energy Consumption (kWh)'].max():.2f}")

st.markdown("---")

# =========================
# INPUT + OUTPUT
# =========================
left, right = st.columns(2)

with left:
    st.subheader("🔢 Enter Details")

    hour = st.slider("Hour", 0, 23, 12)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day of Week", 0, 6, 3)
    weekend = st.selectbox("Weekend?", [0,1])

    temp = st.slider("Temperature (°C)", -10.0, 45.0, 25.0)
    house = st.slider("Household Size", 1, 6, 3)

    appliance = st.selectbox("Appliance Type", df['Appliance Type'].unique())
    season = st.selectbox("Season", df['Season'].unique())

with right:
    st.subheader("⚡ Prediction")

    if st.button("Predict Energy"):

        input_data = np.array([[hour, month, day, weekend,
                                temp, appliance, season, house]])

        prediction = model.predict(input_data)[0]

        st.success(f"⚡ {prediction:.2f} kWh predicted")

        # Visual indicator
        st.progress(min(prediction / df['Energy Consumption (kWh)'].max(), 1.0))

        if prediction > df['Energy Consumption (kWh)'].mean():
            st.warning("⚠️ High energy usage expected")
        else:
            st.info("✅ Energy usage is normal")

st.markdown("---")

# =========================
# OPTIONAL DASHBOARD (FAST)
# =========================
if st.checkbox("📊 Show Analysis Dashboard"):

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Energy Trend")
        st.line_chart(df['Energy Consumption (kWh)'][:200])

    with colB:
        st.subheader("🔥 Appliance Usage")
        st.bar_chart(df.groupby('Appliance Type')['Energy Consumption (kWh)'].sum())

    colC, colD = st.columns(2)

    with colC:
        st.subheader("⏰ Hourly Pattern")
        st.line_chart(df.groupby('Hour')['Energy Consumption (kWh)'].mean())

    with colD:
        st.subheader("🌡️ Temperature Impact")
        st.scatter_chart(df[['Outdoor Temperature (°C)', 'Energy Consumption (kWh)']])

st.markdown("---")

# =========================
# SMART INSIGHTS
# =========================
st.subheader("🧠 Smart Insights")

peak_hour = df.groupby('Hour')['Energy Consumption (kWh)'].mean().idxmax()

st.write(f"🔥 Peak usage hour: **{peak_hour}**")
st.write("💡 Use heavy appliances during off-peak hours")
st.write("🌡️ Optimize temperature settings")
st.write("🔌 Turn off unused devices")

st.success("✅ System Ready | Fast & Optimized")

import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Title of the Web App
st.title("🏎️ ACC Setup Optimizer")

# Track and Car Selection
track = st.selectbox("Select Track", ["Monza", "Spa", "Silverstone"])
car = st.selectbox("Select Car", ["Ferrari 488 GT3", "Porsche 911 GT3", "Lamborghini Huracán GT3"])

# File uploader for telemetry CSV
uploaded_file = st.file_uploader("📂 Upload Telemetry Data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Telemetry Data Preview:", df.head())

    # Feature selection
    features = df[['Speed', 'Throttle', 'Brake', 'SteeringAngle', 'TireTemp_FL', 'TireTemp_FR', 'TireTemp_RL', 'TireTemp_RR']]
    target = df['LapTime']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Show model accuracy
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    st.success(f"✅ Model Trained! Mean Absolute Error: {error:.3f} seconds")

    # Upload ACC Setup File (JSON)
    setup_file = st.file_uploader("📂 Upload ACC Setup File (JSON)", type=["json"])
    if setup_file:
        setup_data = json.load(setup_file)
        st.write("🔧 Current Setup:", setup_data)

        # Optimize setup (adjust multiple parameters)
        suggested_changes = model.predict([df.mean().values])
        setup_data['setup']['tyrePressure'] = list(np.clip(suggested_changes[:4], 25, 32))
        setup_data['setup']['camber'] = list(np.clip(suggested_changes[4:8], -4.0, 0.0))
        setup_data['setup']['toe'] = list(np.clip(suggested_changes[8:12], -0.5, 0.5))

        # Display optimized setup
        st.json(setup_data)

        # Provide a download link
        st.download_button(
            label="💾 Download Optimized Setup",
            data=json.dumps(setup_data, indent=4),
            file_name="optimized_setup.json",
            mime="application/json"
        )

    # Tire Temperature Visualization
    st.subheader("📊 Tire Temperature Visualization")
    fig, ax = plt.subplots()
    ax.plot(df["LapTime"], df["TireTemp_FL"], label="Front Left")
    ax.plot(df["LapTime"], df["TireTemp_FR"], label="Front Right")
    ax.plot(df["LapTime"], df["TireTemp_RL"], label="Rear Left")
    ax.plot(df["LapTime"], df["TireTemp_RR"], label="Rear Right")
    ax.legend()
    st.pyplot(fig)

    # Session State for Saving and Loading Setups
    if "saved_setup" not in st.session_state:
        st.session_state.saved_setup = {}

    if st.button("💾 Save Setup"):
        st.session_state.saved_setup.update(setup_data)
        st.success("Setup Saved Successfully!")

    if st.session_state.saved_setup:
        st.subheader("📁 Saved Setup")
        st.json(st.session_state.saved_setup)

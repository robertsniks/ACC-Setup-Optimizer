import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Title of the Web App
st.title("🏎️ ACC Setup Optimizer for Monza")

# File uploader for telemetry CSV
uploaded_file = st.file_uploader("📂 Upload Telemetry Data (CSV)", type=["csv"])

if uploaded_file:
    # Load telemetry data
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

        # Optimize setup (adjust tire pressures)
        suggested_changes = model.predict([df.mean().values])
        setup_data['setup']['tyrePressure'] = list(np.clip(suggested_changes[:4], 25, 32))

        # Display optimized setup
        st.json(setup_data)

        # Provide a download link
        st.download_button(
            label="💾 Download Optimized Setup",
            data=json.dumps(setup_data, indent=4),
            file_name="optimized_setup.json",
            mime="application/json"
        )

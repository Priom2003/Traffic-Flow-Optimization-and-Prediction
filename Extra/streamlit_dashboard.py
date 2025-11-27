import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="Traffic Prediction Dashboard", layout="wide")
st.title("🚦 Traffic Flow Prediction Dashboard")

# --- Sidebar ---
st.sidebar.header("Dashboard Options")
mode = st.sidebar.radio("Select Mode", ["CSV Predictions", "Live Model Prediction"])

# --- Load Data ---
if mode == "CSV Predictions":
    st.subheader("Displaying Pre-computed Predictions")
    df = pd.read_csv("phase4_predictions.csv")

elif mode == "Live Model Prediction":
    st.subheader("Live Prediction using Saved Models")
    st.info("Will use RF / XGBoost / LSTM to predict from CSV input")
    
    # --- Load Models ---
    try:
        rf_model = joblib.load("rf_model.pkl")
        xgb_model = joblib.load("xgb_model.pkl")
        lstm_model = load_model("lstm_model.h5")
    except Exception as e:
        st.warning(f"Could not load models: {e}. Demo predictions will be random.")

    # --- Load feature CSV (without target labels) ---
    try:
        df = pd.read_csv("phase3_structured_dataset.csv")
    except FileNotFoundError:
        st.error("CSV file not found. Please provide 'phase3_structured_dataset.csv'.")
        st.stop()

    # --- Prepare Features ---
    embedding_cols = [f"emb_{i}" for i in range(384)]
    structured_cols = ['sentiment_numeric', 'sentiment_score','cluster_label', 
                       'latitude', 'longitude', 'hour', 'day', 'weekday', 'month']
    train_cols = embedding_cols + structured_cols  # 393 features

    X_df = df.copy()

    # Add missing columns with zeros
    for col in train_cols:
        if col not in X_df.columns:
            X_df[col] = 0

    # Reorder columns exactly like training
    X_df = X_df[train_cols]

    # Convert to numpy array
    X = X_df.values

    # --- Standardize ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Reshape for LSTM ---
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # --- Predictions ---
    try:
        df["RF_Prediction"] = rf_model.predict(X_scaled)
        df["XGB_Prediction"] = xgb_model.predict(X_scaled)
        lstm_pred_probs = lstm_model.predict(X_lstm, verbose=0)
        df["LSTM_Prediction"] = np.argmax(lstm_pred_probs, axis=1)
    except Exception:
        # Fallback for demo
        num_samples = X_scaled.shape[0]
        df["RF_Prediction"] = np.random.randint(0, 3, size=num_samples)
        df["XGB_Prediction"] = np.random.randint(0, 3, size=num_samples)
        df["LSTM_Prediction"] = np.random.randint(0, 3, size=num_samples)
        st.info("Using simulated predictions for demo purposes.")

    st.success("Live predictions generated!")
    st.dataframe(df.head(10))

# --- Sidebar Filters & Location Handling ---
if "Location" not in df.columns:
    # Add dummy Location column for demo if missing
    df["Location"] = np.random.choice(["Downtown", "Airport", "Highway"], size=len(df))

selected_location = st.sidebar.selectbox(
    "Select Location",
    options=["All"] + list(df["Location"].unique())
)

df_filtered = df.copy()
if selected_location != "All":
    df_filtered = df_filtered[df_filtered["Location"] == selected_location]

# --- Display Table ---
st.subheader("Predicted Traffic Data")
st.dataframe(df_filtered)

# --- Bar Chart: Congestion Levels ---
if mode == "CSV Predictions":
    congestion_col = "Predicted_Congestion"
else:
    congestion_col = "RF_Prediction"  # or "XGB_Prediction" / "LSTM_Prediction"

fig_bar = px.histogram(
    df_filtered,
    x=congestion_col,
    color=congestion_col,
    title="Congestion Level Distribution",
    labels={congestion_col: "Congestion Level"}
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Traffic Density Line Chart ---
if "Traffic_Density_Score" in df_filtered.columns:
    df_filtered["Hour"] = pd.to_datetime(df_filtered["Timestamp"]).dt.hour
    fig_line = px.line(
        df_filtered.groupby("Hour")["Traffic_Density_Score"].mean().reset_index(),
        x="Hour",
        y="Traffic_Density_Score",
        title="Average Traffic Density by Hour",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- Map ---
if "Latitude" in df_filtered.columns and "Longitude" in df_filtered.columns:
    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        color=congestion_col,
        size="sentiment_score" if "sentiment_score" in df_filtered.columns else None,
        hover_name="Timestamp",
        hover_data=["Location"],
        mapbox_style="open-street-map",
        title="Traffic Map"
    )
    st.plotly_chart(fig_map, use_container_width=True)

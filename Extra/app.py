# app.py
# Streamlit dashboard: Hybrid LLM (all-MiniLM-L6-v2) -> LSTM predictor
import streamlit as st
st.set_page_config(page_title="LLM+LSTM Traffic Predictor", layout="wide", initial_sidebar_state="expanded")

import os, time, json
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# Optional shap import (may be slow)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -------------------------
# Helper / caching loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_lstm_model(path="models/best_hybrid_lstm_llm.h5"):
    if not os.path.exists(path):
        st.error(f"LSTM model not found at {path}. Update path in code or place model there.")
        raise FileNotFoundError
    model = load_model(path, compile=False)
    return model

# Optional: if you have a scaler/imputer saved for structured fields
@st.cache_resource(show_spinner=False)
def load_scaler(path="models/fusion_scaler.pkl"):
    if os.path.exists(path):
        return load(path)
    return None

# -------------------------
# UI: Sidebar
# -------------------------
st.sidebar.title("Settings & Controls")
st.sidebar.markdown("Model & runtime controls")

MODEL_PATH = st.sidebar.text_input("LSTM model path", value="models/best_hybrid_lstm_llm.h5")
EMBED_MODEL = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")
RELOAD_MODELS = st.sidebar.button("Reload models")

EXPLAIN_SHAP = st.sidebar.checkbox("Enable SHAP explainability (slow)", value=False)
SHOW_MAP = st.sidebar.checkbox("Show map (if lat/lon present)", value=True)
DOWNLOAD_PATH = st.sidebar.text_input("Output folder", value="outputs")

# load resources (cache)
embedder = load_embedder(EMBED_MODEL)
lstm_model = load_lstm_model(MODEL_PATH)
scaler = load_scaler("models/fusion_scaler.pkl")

# Display quick model info
st.sidebar.markdown("### Model Info")
st.sidebar.write(f"Embedder: **{EMBED_MODEL}**")
st.sidebar.write(f"LSTM model: **{os.path.basename(MODEL_PATH)}**")
try:
    st.sidebar.write("Input shape: " + str(lstm_model.input_shape))
    st.sidebar.write("Output shape: " + str(lstm_model.output_shape))
except Exception:
    pass

# -------------------------
# Layout: header
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.title("🚦 LLM + LSTM Traffic Congestion Predictor")
    st.markdown("Predict traffic congestion level from tweets using **MiniLM** embeddings + **LSTM**.")
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Traffic_Congestion.jpg/320px-Traffic_Congestion.jpg", width=140)

st.markdown("---")

# -------------------------
# TAB: Single prediction / Batch / Analysis
# -------------------------
tabs = st.tabs(["Single Tweet", "Batch CSV", "Visualize / Map", "Model Eval & SHAP", "Logs & Export"])

# -------------------------
# Tab 1: Single Tweet
# -------------------------
with tabs[0]:
    st.header("Single tweet prediction")
    tweet_text = st.text_area("Enter a tweet (or paste example):", height=120,
                              placeholder="Traffic jam on CMC road near Katpadi...")

    col_a, col_b = st.columns([1,1])
    with col_a:
        weight = st.slider("Confidence threshold (show only if prob >=)", 0.0, 1.0, 0.3, 0.05)

    with col_b:
        if st.button("Predict Tweet"):
            if tweet_text.strip() == "":
                st.warning("Please enter a tweet.")
            else:
                with st.spinner("Embedding and predicting..."):
                    emb = embedder.encode([tweet_text])  # shape (1, embed_dim)
                    X = np.expand_dims(emb, axis=1)     # (1, 1, embed_dim)
                    probs = lstm_model.predict(X, verbose=0)[0]
                    labels = ["Low Congestion", "Medium Congestion", "High Congestion"]
                    pred_idx = int(np.argmax(probs))
                    pred_label = labels[pred_idx]
                    st.success(f"Prediction: **{pred_label}** (p = {probs[pred_idx]:.3f})")

                    # probability bar chart
                    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=['green','orange','red']))
                    fig.update_layout(title="Predicted probabilities", yaxis_range=[0,1])
                    st.plotly_chart(fig, use_container_width=True)

                    # SHAP for single input (optional)
                    if EXPLAIN_SHAP and SHAP_AVAILABLE:
                        st.info("Computing SHAP (may take a few seconds)...")
                        try:
                            # use KernelExplainer for TF LSTM compatibility
                            # prepare background as small subset of train (not available here) -> use embedder on sample
                            background = embedder.encode(["traffic", "jam", "clear road", "accident", "heavy traffic"])
                            bkg = np.expand_dims(background, axis=1)
                            f = lambda x: lstm_model.predict(x.reshape(x.shape[0], 1, -1))
                            explainer = shap.KernelExplainer(f, background)
                            shap_vals = explainer.shap_values(emb, nsamples=100)
                            st.pyplot(shap.summary_plot(shap_vals, emb, feature_names=[f"emb_{i}" for i in range(emb.shape[1])], show=False))
                        except Exception as e:
                            st.error("SHAP failed: " + str(e))

# -------------------------
# Tab 2: Batch CSV
# -------------------------
with tabs[1]:
    st.header("Batch prediction from CSV")
    st.markdown("Upload a CSV with column `raw_text_tweet`. Optionally include `latitude`,`longitude`,`created_at`,`actual_label`.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df_in.head(5))

        if "raw_text_tweet" not in df_in.columns:
            st.error("CSV must contain 'raw_text_tweet' column.")
        else:
            batch_size = st.number_input("Batch size (embedding/predict chunk)", min_value=16, max_value=1024, value=128)
            if st.button("Run batch prediction"):
                with st.spinner("Embedding and predicting (this may take time)..."):
                    texts = df_in["raw_text_tweet"].astype(str).tolist()
                    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
                    X_all = np.expand_dims(embeddings, axis=1)
                    probs = lstm_model.predict(X_all, verbose=0)
                    preds = np.argmax(probs, axis=1)
                    df_in["predicted_label"] = preds
                    df_in["predicted_text"] = df_in["predicted_label"].map({0:"Low",1:"Medium",2:"High"})
                    df_in["prob_low"] = probs[:,0]; df_in["prob_medium"] = probs[:,1]; df_in["prob_high"] = probs[:,2]

                    st.success("Batch prediction completed.")
                    st.dataframe(df_in.head(10))

                    # offer download
                    out_path = os.path.join(DOWNLOAD_PATH if os.path.exists(DOWNLOAD_PATH) else ".", f"predictions_{int(time.time())}.csv")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    df_in.to_csv(out_path, index=False)
                    st.download_button("Download predictions CSV", data=open(out_path, "rb"), file_name=os.path.basename(out_path))

# -------------------------
# Tab 3: Visualize / Map
# -------------------------
with tabs[2]:
    st.header("Spatial & Temporal Visualization")
    st.markdown("Use last uploaded batch or sample CSV to visualize location and hourly trends.")
    if 'df_in' not in locals():
        st.info("Run a batch prediction first in 'Batch CSV' tab to visualize map/time charts.")
    else:
        df_vis = df_in.copy()
        if "latitude" in df_vis.columns and "longitude" in df_vis.columns and SHOW_MAP:
            st.subheader("Predicted congestion on map")
            # map uses predicted_label (0/1/2) to color
            df_vis['pred_text'] = df_vis['predicted_text']
            fig_map = px.scatter_mapbox(df_vis, lat="latitude", lon="longitude", color="pred_text",
                                       hover_name="raw_text_tweet", zoom=11, height=600,
                                       color_discrete_map={"Low":"green","Medium":"orange","High":"red"})
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Latitude/Longitude missing in the batch. Map not shown.")

        if "created_at" in df_vis.columns:
            st.subheader("Hourly predicted counts")
            df_vis['created_at'] = pd.to_datetime(df_vis['created_at'])
            df_vis["hour"] = df_vis["created_at"].dt.hour
            agg = df_vis.groupby("hour")['predicted_label'].value_counts().unstack(fill_value=0)
            st.bar_chart(agg)

# -------------------------
# Tab 4: Model Eval & SHAP
# -------------------------
with tabs[3]:
    st.header("Model Evaluation & Explainability")
    st.markdown("If your CSV included an `actual_label` or `traffic_label_numeric` column we compute metrics.")

    if uploaded is None:
        st.info("Upload and run batch prediction first in the 'Batch CSV' tab to enable evaluation.")
    else:
        if 'actual_label' in df_in.columns or 'traffic_label_numeric' in df_in.columns:
            gt_col = 'actual_label' if 'actual_label' in df_in.columns else 'traffic_label_numeric'
            y_true = df_in[gt_col].values
            y_pred = df_in['predicted_label'].values
            st.write("Classification report:")
            st.text(classification_report(y_true, y_pred, target_names=["Low","Medium","High"]))
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=["Low","Medium","High"], y=["Low","Medium","High"], color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("No ground truth column found in uploaded CSV.")

    st.markdown("---")
    st.subheader("SHAP Explainability (KernelExplainer fallback)")
    if EXPLAIN_SHAP:
        if not SHAP_AVAILABLE:
            st.error("SHAP package not installed or failed to import. Install `shap` to enable.")
        else:
            try:
                with st.spinner("Preparing SHAP (this may be slow)..."):
                    # Prepare small background (sample from uploaded or from default)
                    if 'df_in' in locals():
                        sample_texts = df_in['raw_text_tweet'].astype(str).sample(min(200, len(df_in)), random_state=42).tolist()
                    else:
                        sample_texts = ["traffic jam", "smooth traffic", "accident on road", "clear road"]
                    background_emb = embedder.encode(sample_texts, show_progress_bar=False)
                    background = background_emb.reshape(background_emb.shape[0], -1)

                    # KernelExplainer needs 2D input - define wrapper
                    f = lambda X_flat: lstm_model.predict(X_flat.reshape(X_flat.shape[0], 1, -1))
                    explainer = shap.KernelExplainer(f, background[:50])
                    # explain a small set only
                    explain_emb = background[:10]
                    shap_vals = explainer.shap_values(explain_emb, nsamples=100)
                    st.pyplot(shap.summary_plot(shap_vals, explain_emb, feature_names=[f"emb_{i}" for i in range(explain_emb.shape[1])], show=False))
            except Exception as e:
                st.error("SHAP compute failed: " + str(e))

# -------------------------
# Tab 5: Logs & Export
# -------------------------
with tabs[4]:
    st.header("Logs, Save & Export")
    st.markdown("Recent predictions (last batch):")
    if 'df_in' in locals():
        st.dataframe(df_in[['raw_text_tweet','predicted_text','prob_high']].head(20))
    else:
        st.info("Run a batch prediction to populate logs.")
    st.markdown("### Save artifacts")
    if st.button("Save model summary"):
        info = {
            "embedder": EMBED_MODEL if 'EMBED_MODEL' in locals() else "all-MiniLM-L6-v2",
            "lstm_model": MODEL_PATH,
            "timestamp": time.ctime()
        }
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)
        with open(os.path.join(DOWNLOAD_PATH,"model_info.json"), "w") as f:
            json.dump(info, f)
        st.success("Saved model_info.json")

st.markdown("---")
st.caption("Dashboard built for demo. For large datasets or production, consider batching embeddings offline and using a faster inference server.")
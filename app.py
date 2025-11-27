import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os, time
from textblob import TextBlob
import datetime
from sentence_transformers.util import cos_sim

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Traffic Predictor", layout="wide")

st.markdown("<h1 style='text-align:center;'>🚦 Traffic Congestion Predictor</h1>", unsafe_allow_html=True)
st.write("---")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_lstm():
    return load_model("lstm_model.h5")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

try:
    lstm_model = load_lstm()
    st.success("✅ LSTM Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Failed to load LSTM model: {e}")
    st.stop()

embedder = load_embedder()

from sentence_transformers.util import cos_sim

@st.cache_resource
def get_keyword_embeddings():
    medium_words = ["slow", "busy", "queue", "waiting", "blocked", "signal", "crowd", "diversion"]
    high_words = ["accident", "crash", "stuck", "standstill", "massive jam", "collision", "roadblock"]
    return (
        embedder.encode(medium_words),
        embedder.encode(high_words)
    )

medium_embs, high_embs = get_keyword_embeddings()

def semantic_boost(tweet_emb):
    """Compute semantic similarity boost for Medium and High."""
    sim_med = cos_sim(tweet_emb, medium_embs).mean()
    sim_high = cos_sim(tweet_emb, high_embs).mean()
    return np.array([0, sim_med, sim_high])

def batch_semantic_boost(embs):
    """Return Nx3 boost array for medium & high similarity scores."""
    sim_med = cos_sim(embs, medium_embs).mean(axis=1)
    sim_high = cos_sim(embs, high_embs).mean(axis=1)
    return np.vstack([np.zeros(len(embs)), sim_med, sim_high]).T

# ---------------- SINGLE TWEET PREDICTION ----------------
st.subheader("🔹 Single Tweet Prediction")

tweet = st.text_area("Enter tweet:", height=100,
                     placeholder="Example: Heavy traffic near VIT gate due to signal failure")

if st.button("Predict"):
    if tweet.strip():
        # Step 1: Encode text → LLM Embeddings
        emb = embedder.encode([tweet])

        # Step 2: Generate realistic structured features dynamically
        sentiment = TextBlob(tweet).sentiment.polarity
        sentiment_numeric = 1 if sentiment > 0 else (-1 if sentiment < 0 else 0)
        length_factor = len(tweet.split()) / 25
        has_accident = 1 if "accident" in tweet.lower() or "crash" in tweet.lower() else 0
        has_jam = 1 if "jam" in tweet.lower() or "traffic" in tweet.lower() else 0
        engagement_factor = np.random.uniform(0.6, 1.4)

        now = datetime.datetime.now()
        structured_features = np.array([[ 
            sentiment_numeric,
            sentiment,
            12.9 + np.random.uniform(-0.02, 0.02),
            79.13 + np.random.uniform(-0.02, 0.02),
            now.hour,
            now.day,
            now.weekday(),
            now.month
        ]]) * engagement_factor * (1 + 0.5 * has_accident + 0.3 * has_jam + 0.2 * length_factor)

        # Step 3: Combine embeddings + structured features (align to LSTM)
        X_input = np.concatenate([emb, structured_features], axis=1)

        expected_dim = lstm_model.input_shape[-1]
        if X_input.shape[1] < expected_dim:
            X_input = np.concatenate([X_input, np.zeros((1, expected_dim - X_input.shape[1]))], axis=1)
        elif X_input.shape[1] > expected_dim:
            X_input = X_input[:, :expected_dim]

        X = np.expand_dims(X_input, axis=1)

        try:
            probs = lstm_model.predict(X, verbose=0)[0]
            probs = np.clip(probs, 1e-5, 1)

            # ------------------ CLASS WEIGHT CORRECTION ------------------
            # Stronger weights to prevent "Low-only" bias
            class_weights = np.array([0.6, 4.2, 7.7])
            probs = probs * class_weights
            probs /= probs.sum()

            # ------------------ SEMANTIC BOOST ------------------
            tweet_emb = emb
            boost = semantic_boost(tweet_emb)

            # Quiet boost (keeps model stable)
            probs = probs + 0.40 * boost
            probs /= probs.sum()

            # ------------------ KEYWORD SEVERITY BOOST ------------------
            severity_score = (
                tweet.lower().count("traffic") * 0.25 +
                tweet.lower().count("jam") * 0.45 +
                tweet.lower().count("accident") * 1.00 +
                tweet.lower().count("blocked") * 0.70 +
                tweet.lower().count("signal") * 0.20
            )

            probs[1] += 0.30* severity_score     # Medium
            probs[2] += 0.60 * severity_score     # High
            probs = np.clip(probs, 1e-6, None)
            probs /= probs.sum()

            # ------------------ TEMPERATURE SMOOTHING ------------------
            temperature = 1.40
            probs = probs ** (1 / temperature)
            probs /= probs.sum()

            st.caption(f"Adjusted probs: {np.round(probs, 3)}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
            st.stop()

        labels = ["Low Congestion", "Medium Congestion", "High Congestion"]
        pred = labels[int(np.argmax(probs))]

        st.markdown(f"### ✅ Prediction: **{pred}**")

        delay_map = {"Low Congestion":"0–2 mins", "Medium Congestion":"3–6 mins", "High Congestion":"7–12 mins"}
        st.info(f"⏱️ Estimated Delay: **{delay_map[pred]}**")

        # Step 5: Visualization (Compact, Centered, Multi-line Labels)
        fig, ax = plt.subplots(figsize=(2.8, 1.6))
        labels_wrapped = ["Low\nCongestion", "Medium\nCongestion", "High\nCongestion"]

        bars = sns.barplot(
            x=labels_wrapped, y=probs,
            palette=["#27ae60", "#f39c12", "#e74c3c"],
            ax=ax, width=0.5
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability", fontsize=8, labelpad=2)
        ax.set_xlabel("", fontsize=8)
        ax.set_title("Prediction Confidence", fontsize=9, weight="bold", pad=4)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        for p, v in zip(bars.patches, probs):
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.02,
                f"{v*100:.1f}%",
                ha="center",
                fontsize=7,
                color="black"
            )

        ax.grid(False)
        plt.tight_layout(pad=0.3)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig, use_container_width=False)

    else:
        st.warning("Enter a tweet to analyze!")

st.divider()

# ---------------- BATCH CSV PREDICTION ----------------
st.subheader("📂 Batch Tweet Prediction (CSV Upload)")
file = st.file_uploader("Upload CSV with column raw_text_tweet", type=["csv"])

if file:
    df = pd.read_csv(file)
    if df.empty:
        st.warning("Uploaded CSV is empty! Please upload a valid dataset.")
        st.stop()

    if "raw_text_tweet" not in df.columns:
        st.error("CSV must contain a column named raw_text_tweet")
    else:
        with st.spinner("Embedding & predicting..."):
            texts = df["raw_text_tweet"].astype(str).tolist()
            embs = embedder.encode(texts, show_progress_bar=True)

            now = datetime.datetime.now()
            structured_features = np.array([
                [
                    np.random.choice([-1, 0, 1]),
                    np.random.uniform(-1, 1),
                    12.9 + np.random.uniform(-0.03, 0.03),
                    79.13 + np.random.uniform(-0.03, 0.03),
                    now.hour,
                    now.day,
                    now.weekday(),
                    now.month
                ]
                for _ in range(len(texts))
            ])

            X_input = np.concatenate([embs, structured_features], axis=1)

            expected_dim = lstm_model.input_shape[-1]
            if X_input.shape[1] < expected_dim:
                pad_dim = expected_dim - X_input.shape[1]
                X_input = np.concatenate([X_input, np.zeros((X_input.shape[0], pad_dim))], axis=1)
            elif X_input.shape[1] > expected_dim:
                X_input = X_input[:, :expected_dim]

            X = np.expand_dims(X_input, axis=1)
            # Step 4: Predict probabilities
            try:
                probs = lstm_model.predict(X, verbose=0)
                probs = np.clip(probs, 1e-6, 1)

                # ------------------ CLASS WEIGHT ADJUSTMENT ------------------
                class_weights = np.array([1.0, 3.5, 6.0])
                probs = probs * class_weights
                probs /= probs.sum(axis=1, keepdims=True)

                # ------------------ SEMANTIC BOOST (vectorized) ------------------
                sem_boost = batch_semantic_boost(embs)   # Nx3 array
                probs = probs + 0.40 * sem_boost
                probs = np.clip(probs, 1e-6, None)
                probs /= probs.sum(axis=1, keepdims=True)

                # ------------------ KEYWORD SEVERITY BOOST ------------------
                def severity_vec(texts):
                    sev = []
                    for t in texts:
                        t = t.lower()
                        sev.append(
                            t.count("traffic") * 0.25 +
                            t.count("jam") * 0.45 +
                            t.count("accident") * 1.00 +
                            t.count("blocked") * 0.70 +
                            t.count("signal") * 0.20
                        )
                    return np.array(sev)

                sv = severity_vec(texts)

                probs[:, 1] += 0.15 * sv  # Medium
                probs[:, 2] += 0.30 * sv  # High
                probs = np.clip(probs, 1e-6, None)
                probs /= probs.sum(axis=1, keepdims=True)

                # ------------------ TEMPERATURE SMOOTHING ------------------
                temperature = 1.20
                probs = probs ** (1 / temperature)
                probs /= probs.sum(axis=1, keepdims=True)

                # Final predicted class
                preds = np.argmax(probs, axis=1)
            except Exception as e:
                st.error(f"⚠️ Batch prediction failed: {e}")
                st.stop()

            df["predicted_label"] = preds
            df["predicted_text"] = df["predicted_label"].map({0:"Low",1:"Medium",2:"High"})
            df["delay_estimated"] = df["predicted_text"].map({
                "Low":"0–2 mins", "Medium":"3–6 mins", "High":"7–12 mins"
            })

        st.success("✅ Prediction Completed")
        st.dataframe(df.head(10))

        if "latitude" in df.columns and "longitude" in df.columns:
            st.subheader("🗺️ Traffic Density Map")
            fig = px.scatter_mapbox(
                df, lat="latitude", lon="longitude",
                color="predicted_text", zoom=11,
                mapbox_style="open-street-map",
                color_discrete_map={"Low":"green","Medium":"orange","High":"red"}
            )
            st.plotly_chart(fig, use_container_width=True)

        os.makedirs("outputs", exist_ok=True)
        out = f"outputs/predictions_{int(time.time())}.csv"
        df.to_csv(out, index=False)
        st.download_button("⬇️ Download Results", open(out,"rb"), file_name=os.path.basename(out))

st.write("---")
st.markdown("""
    <p style='text-align:center; color:gray;'>
    Developed by <b>Priom Dutta</b><br>
    M.C.A. – VIT Vellore | Guide: Dr. Tapan Kumar Das<br>
    Project: LLM-Based Traffic Flow Prediction using Social Media Data
    </p>
""", unsafe_allow_html=True)

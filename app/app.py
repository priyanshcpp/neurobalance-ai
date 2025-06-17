import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Page setup
st.set_page_config(page_title="NeuroBalance AI", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #f14e8c;'>🧠 NeuroBalance AI</h1>"
    "<h4 style='text-align: center; color: #e0e0e0;'>Decode your dosha balance with AI + Ayurveda</h4>",
    unsafe_allow_html=True
)

# Load + train model from CSV
@st.cache_data
def train_models():
    df = pd.read_csv("data/dosha_data.csv")
    df = pd.get_dummies(df, columns=["region", "temp_feel"], drop_first=True)

    X = df.drop(columns=["vata", "pitta", "kapha"])
    y_vata = df["vata"]
    y_pitta = df["pitta"]
    y_kapha = df["kapha"]

    X_train, _, yv_train, _ = train_test_split(X, y_vata, test_size=0.2)
    _, _, yp_train, _ = train_test_split(X, y_pitta, test_size=0.2)
    _, _, yk_train, _ = train_test_split(X, y_kapha, test_size=0.2)

    vata_model = LinearRegression().fit(X_train, yv_train)
    pitta_model = LinearRegression().fit(X_train, yp_train)
    kapha_model = LinearRegression().fit(X_train, yk_train)

    return vata_model, pitta_model, kapha_model, X.columns.tolist()

model_vata, model_pitta, model_kapha, features = train_models()

# Input
st.markdown("### 🛌 Sleep & Daily Rhythm")
sleep = st.slider("🛏️ Sleep Hours", 4.0, 10.0, 7.0)
wake_time = st.slider("⏰ Wake-up Time (24hr)", 3, 10, 6)

st.markdown("### 😖 Stress & Hydration")
stress = st.slider("😤 Stress Level", 1, 10, 5)
water = st.slider("💧 Water Intake (Liters)", 1.0, 5.0, 2.5)

st.markdown("### 🍽️ Diet & Activity")
meal = st.slider("🍲 Meal Heaviness", 1, 10, 5)
exercise = st.slider("🏃 Exercise Days per Week", 0, 7, 3)

st.markdown("### 🌡️ Environment")
region = st.selectbox("🌍 Climate Region", ["hot", "cold", "humid"])
temp_feel = st.selectbox("🌡️ Body Temperature Sensitivity", ["cold", "warm"])

region_hot = 1 if region == "hot" else 0
region_humid = 1 if region == "humid" else 0
temp_feel_warm = 1 if temp_feel == "warm" else 0

x = np.array([sleep, wake_time, stress, water, meal, exercise,
              region_hot, region_humid, temp_feel_warm]).reshape(1, -1)

# Prediction
if st.button("🔮 Predict Wellness Profile"):
    vata = model_vata.predict(x)[0]
    pitta = model_pitta.predict(x)[0]
    kapha = model_kapha.predict(x)[0]

    st.markdown("## 🧘 Your Wellness Balance")

    st.markdown(f"🌬️ **Vata (Air / Movement)** – `{vata:.2f}%`")
    st.progress(int(vata))

    st.markdown(f"🔥 **Pitta (Fire / Metabolism)** – `{pitta:.2f}%`")
    st.progress(int(pitta))

    st.markdown(f"🌍 **Kapha (Earth / Stability)** – `{kapha:.2f}%`")
    st.progress(int(kapha))

    st.success("✅ Tip: Adjust your lifestyle to maintain balance across doshas.")

# Sidebar
st.sidebar.markdown("## 💡 About This App")
st.sidebar.markdown("""
This is a mind–body intelligence tool powered by machine learning  
and inspired by ancient Ayurvedic wisdom.

### 🌿 Dosha Types
- **Vata (Air)** → Movement, creativity, quickness  
- **Pitta (Fire)** → Energy, digestion, leadership  
- **Kapha (Earth)** → Calmness, strength, grounding  
""")

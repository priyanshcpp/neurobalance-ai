import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(page_title="NeuroBalance AI", layout="centered")

# ✅ Load models
model_vata = joblib.load("model/vata_model.pkl")
model_pitta = joblib.load("model/pitta_model.pkl")
model_kapha = joblib.load("model/kapha_model.pkl")
features = joblib.load("model/features.pkl")  # Optional

# Styled title
st.markdown(
    "<h1 style='text-align: center; color: #f14e8c;'>🧠 NeuroBalance AI</h1>"
    "<h4 style='text-align: center; color: #e0e0e0;'>Decode your dosha balance with AI + Ayurveda</h4>",
    unsafe_allow_html=True
)


# Inputs
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

# One-hot encode
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

    st.markdown("---")
    st.caption("✨ Based on Ayurvedic principles and your personalized inputs.")

# Sidebar – About
st.sidebar.markdown("## 💡 About This App")
st.sidebar.markdown("""
NeuroBalance AI is a mind–body intelligence tool powered by machine learning  
and inspired by ancient Ayurvedic wisdom.  
It helps you discover your internal balance through daily lifestyle patterns.

### 🌿 Dosha Types
- **Vata (Air)** → Movement, creativity, quickness  
- **Pitta (Fire)** → Energy, digestion, leadership  
- **Kapha (Earth)** → Calmness, strength, grounding  
""")

# Dosha info section
st.markdown("### 🧾 Dosha Insights")
st.markdown("""
- **🌬️ Vata** → Creativity, energy flow, but excess causes anxiety.
- **🔥 Pitta** → Sharp mind and digestion, excess causes anger or burnout.
- **🌍 Kapha** → Calm and stable, but excess can lead to lethargy.
""")

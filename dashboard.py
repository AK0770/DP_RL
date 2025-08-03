import streamlit as st
import requests

st.title("🧠 Dynamic Pricing Dashboard")

step = st.slider("Current Day (Step)", min_value=0, max_value=30, value=0)
inventory = st.slider("Inventory Level", min_value=0, max_value=500, value=250)
demand_factor = st.slider("Demand Factor", min_value=0.5, max_value=1.5, value=1.0, step=0.01)

if st.button("📈 Get Price Recommendation"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"step": step, "inventory": inventory, "demand_factor": demand_factor},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        st.success(f"💰 Recommended Price: **${result['recommended_price']}**")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Prediction error: {e}")

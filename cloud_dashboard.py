import streamlit as st
import requests
import time

# ======================================================
# CONFIG
# ======================================================
API_URL = "https://hazard-detection-of-chandrayaan-3-imagery.onrender.com/latest"

st.set_page_config(
    page_title="🛰️ Chandrayaan-3 Ground Station",
    layout="centered"
)

st.title("🛰️ Chandrayaan-3 Ground Station")
st.caption("Live telemetry from rover (Edge AI)")

# ======================================================
# FETCH DATA
# ======================================================
def fetch_latest():
    try:
        r = requests.get(API_URL, timeout=3)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

data = fetch_latest()

# ======================================================
# UI
# ======================================================
if data is None:
    st.error("📴 No signal from rover")
    st.write("Waiting for next uplink…")
else:
    st.success("☁️ Signal received from rover")

    st.markdown(f"**🕒 Timestamp:** {data['timestamp']}")
    st.markdown(f"**📡 Source:** {data['source']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Safe Zone (%)", f"{data['safe']:.1f}")
    c2.metric("Rocks (%)", f"{data['rocks']:.1f}")
    c3.metric("Crater (%)", f"{data['crater']:.1f}")

    hazard = data["rocks"] + data["crater"]

    st.divider()

    if data["safe"] > 80:
        st.success("✅ MISSION STATUS: SAFE FOR NAVIGATION")
    elif data["safe"] > 60:
        st.warning("⚠️ MISSION STATUS: PROCEED WITH CAUTION")
    else:
        st.error("🚫 MISSION STATUS: HAZARDOUS TERRAIN")

# ======================================================
# AUTO REFRESH
# ======================================================
st.divider()
st.caption("Auto-refreshes every 10 seconds")
time.sleep(10)
st.rerun()

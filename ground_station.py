import streamlit as st
import requests

st.title("🛰️ Chandrayaan-3 Ground Station")

try:
    data = requests.get(
        "https://hazard-detection-of-chandrayaan-3-imagery.onrender.com/latest",
        timeout=2
    ).json()

    if data:
        st.success("☁️ Latest rover data received")
        st.write("Timestamp:", data["timestamp"])
        st.write("Source:", data["source"])
        st.metric("Safe Zone (%)", data["safe"])
        st.metric("Rocks (%)", data["rocks"])
        st.metric("Crater (%)", data["crater"])
    else:
        st.warning("Waiting for rover sync…")

except:
    st.error("Cloud unreachable")

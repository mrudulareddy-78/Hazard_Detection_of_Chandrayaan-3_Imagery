import streamlit as st

st.set_page_config(page_title="Ground Station", layout="centered")

st.title("🛰️ Chandrayaan-3 Ground Station")

if "last_data" not in st.session_state:
    st.session_state.last_data = None

import json
import streamlit as st

st.title("🛰️ Chandrayaan-3 Ground Station")

try:
    with open("sync_data.json") as f:
        data = json.load(f)

    if data:
        st.success("☁️ Latest rover data received")
        st.write("🕒 Timestamp:", data["timestamp"])
        st.write("📡 Source:", data["source"])

        st.metric("Safe Zone (%)", data["safe"])
        st.metric("Rocks (%)", data["rocks"])
        st.metric("Crater (%)", data["crater"])
    else:
        st.warning("📴 No data received yet")

except:
    st.warning("📴 No data received yet")
    st.write("Waiting for rover sync…")


# UI
if st.session_state.last_data:
    d = st.session_state.last_data
    st.success("☁️ Latest rover data received")

    st.write("🕒 Timestamp:", d["timestamp"])
    st.write("📡 Source:", d["source"])

    st.metric("Safe Zone (%)", f"{d['safe']:.1f}")
    st.metric("Rocks (%)", f"{d['rocks']:.1f}")
    st.metric("Crater (%)", f"{d['crater']:.1f}")

else:
    st.warning("📴 No data received yet")
    st.write("Waiting for rover sync…")

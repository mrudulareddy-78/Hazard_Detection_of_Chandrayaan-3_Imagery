import streamlit as st

st.set_page_config(page_title="Ground Station", layout="centered")

st.title("🛰️ Chandrayaan-3 Ground Station")

if "last_data" not in st.session_state:
    st.session_state.last_data = None

# API endpoint (used by edge)
if st.query_params.get("api") == "update":
    data = st.json()
    st.session_state.last_data = data
    st.write({"status": "ok"})
    st.stop()

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

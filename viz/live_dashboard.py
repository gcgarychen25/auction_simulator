import json
import time
import threading

import plotly.graph_objects as go
import streamlit as st
from websockets.sync.client import connect

st.set_page_config(layout="wide")

chart_placeholder = st.empty()
log_placeholder = st.empty()

EVENTS = []
PRICES = []
TIMES = []


def listener():
    try:
        with connect("ws://localhost:8765") as ws:
            for message in ws:
                data = json.loads(message)
                EVENTS.append(data)
                if data.get("type") == "bid":
                    PRICES.append(data["payload"]["price"])
                    TIMES.append(data["ts"])
                st.experimental_rerun()
    except Exception:
        time.sleep(1)
        st.experimental_rerun()


if "_listener" not in st.session_state:
    thread = threading.Thread(target=listener, daemon=True)
    thread.start()
    st.session_state["_listener"] = True

if PRICES:
    fig = go.Figure(go.Scatter(x=TIMES, y=PRICES, mode="lines+markers"))
    chart_placeholder.plotly_chart(fig, use_container_width=True)
else:
    chart_placeholder.write("Waiting for events...")

if EVENTS:
    lines = [f"{e['actor']}: {e['type']} {e['payload']}" for e in EVENTS[-25:]]
    log_placeholder.markdown("\n".join(lines))

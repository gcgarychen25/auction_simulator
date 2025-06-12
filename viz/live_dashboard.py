import streamlit as st
import plotly.graph_objs as go
import json
import time
import os
from collections import deque
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Live Auction Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- Constants ---
LIVE_LOG_FILE = "live_auction.log"

# --- State Initialization ---
if 'events' not in st.session_state:
    st.session_state.events = deque(maxlen=200)
if 'lines_read' not in st.session_state:
    st.session_state.lines_read = 0

# --- File Reader ---
def read_new_events():
    """Reads new lines from the log file since the last read."""
    if not os.path.exists(LIVE_LOG_FILE):
        return
    
    with open(LIVE_LOG_FILE, 'r') as f:
        # Go to the start of the last line read
        for _ in range(st.session_state.lines_read):
            try:
                next(f)
            except StopIteration:
                return # No new lines
        
        # Read new lines
        new_lines = f.readlines()
        for line in new_lines:
            try:
                event = json.loads(line)
                st.session_state.events.appendleft(event)
                st.session_state.lines_read += 1
            except json.JSONDecodeError:
                continue # Ignore malformed lines

# --- UI Rendering ---
st.title("🏠 Live Auction Dashboard")
st.caption(f"Watching log file: `{LIVE_LOG_FILE}`")

# Read new data on each run
read_new_events()

if not st.session_state.events:
    st.info("Waiting for the first event from the auction simulator...")
    st.write("If the simulation is running, events should appear here shortly.")
else:
    # --- Data Processing ---
    prices = []
    timestamps = []
    for event in reversed(st.session_state.events):
        if event.get('type') == 'bid' and 'amount' in event.get('payload', {}):
            prices.append(event['payload']['amount'])
            timestamps.append(pd.to_datetime(event['ts'], unit='s'))

    # --- Layout ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Live Agent Transcript")
        transcript_container = st.container(height=500)
        for event in st.session_state.events: # Newest first
            actor = event.get('actor', 'System')
            event_type = event.get('type', 'unknown').upper()
            payload = event.get('payload', {})
            ts = pd.to_datetime(event['ts'], unit='s').strftime('%H:%M:%S')

            with transcript_container:
                if event_type == "BID":
                    st.markdown(f"**{ts}** | **:moneybag: BID** | `{actor}` bids **${payload.get('amount', 0):,.2f}**")
                elif event_type == "ASK":
                    st.markdown(f"**{ts}** | **:question: ASK** | `{actor}`: *{payload.get('question', 'N/A')}*")
                elif event_type == "ANSWER":
                    st.markdown(f"**{ts}** | **:speech_balloon: ANSWER** | `Seller`: *{payload.get('answer', 'N/A')}*")
                elif event_type == "AUCTION_START":
                    st.success(f"**{ts}** | **🎉 AUCTION STARTED**")
                elif event_type == "AUCTION_END":
                    st.success(f"**{ts}** | **🏁 AUCTION ENDED! Winner: `{payload.get('winner', 'N/A')}` at `${payload.get('final_price', 0):,.2f}`**")
                # Add other event types as needed

    with col2:
        st.subheader("📈 Live Price Chart")
        if prices:
            chart_data = pd.DataFrame({'Time': timestamps, 'Price': prices}).set_index('Time')
            st.line_chart(chart_data)
        else:
            st.info("No bids have been placed yet.")

# --- Auto-refresh ---
time.sleep(1)
st.rerun() 
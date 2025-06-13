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
    """Reads new lines from the log file, appending them to the event deque."""
    if not os.path.exists(LIVE_LOG_FILE):
        return
    
    with open(LIVE_LOG_FILE, 'r') as f:
        # Fast-forward to the last read line
        for _ in range(st.session_state.lines_read):
            try:
                next(f)
            except StopIteration:
                return
        
        # Read all new lines
        for line in f:
            try:
                event = json.loads(line)
                # Add new events to the right, for chronological order
                st.session_state.events.append(event)
                st.session_state.lines_read += 1
            except json.JSONDecodeError:
                continue

def get_short_name(actor_id: str) -> str:
    """Creates a short, 2-character name for the chat avatar."""
    if "B1" in actor_id: return "B1"
    if "B2" in actor_id: return "B2"
    if "B3" in actor_id: return "B3"
    if "B4" in actor_id: return "B4"
    if "B5" in actor_id: return "B5"
    if "Seller" in actor_id: return "S"
    return "SYS"

def get_avatar(actor_id: str) -> str:
    """Returns an emoji avatar for the actor."""
    if "Seller" in actor_id:
        return "👨‍⚖️"
    if "B" in actor_id:  # Any buyer
        return "👤"
    return "🤖"  # System

# --- UI Rendering ---
st.title("🏠 Live Auction Dashboard")
st.caption(f"Watching log file: `{LIVE_LOG_FILE}`")

read_new_events()

if not st.session_state.events:
    st.info("Waiting for the first event from the auction simulator...")
else:
    # --- Data Processing ---
    bids = [
        (pd.to_datetime(e['ts'], unit='s'), e['payload']['amount'])
        for e in st.session_state.events
        if e.get('type') == 'bid' and 'amount' in e.get('payload', {})
    ]
    
    # --- Layout ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Live Agent Transcript")
        transcript_container = st.container(height=500, border=True)
        for event in st.session_state.events:
            actor = event.get('actor', 'System')
            event_type = event.get('type', 'unknown').upper()
            payload = event.get('payload', {})
            short_name = get_short_name(actor)
            avatar = get_avatar(actor)

            with transcript_container:
                if event_type == "AUCTION_START":
                    st.info("🎉 Auction Started!")
                elif event_type == "BID":
                    with st.chat_message(name=short_name, avatar="💰"):
                        st.markdown(f"**{actor}**")
                        bid_amount = payload.get('bid', 0)
                        st.markdown(f"I bid **${bid_amount:,.2f}**")
                
                elif event_type == "QA_PAIR":
                    # Render the buyer's question
                    with st.chat_message(name=short_name, avatar=avatar):
                        st.markdown(f"**{actor}**")
                        st.markdown(f"*{payload.get('question', 'No question provided')}*")
                    # Render the seller's answer immediately after
                    with st.chat_message(name="Seller", avatar="👨‍⚖️"):
                        st.markdown(f"**Seller**")
                        st.markdown(f"*{payload.get('answer', 'No answer provided')}*")

                elif event_type == "FOLD":
                    with st.chat_message(name=short_name, avatar=avatar):
                        st.markdown(f"**{actor}** folds.")
                elif event_type == "AUCTION_END":
                    st.success(f"🏁 Auction Ended! Winner: `{payload.get('winner', 'N/A')}` at `${payload.get('final_price', 0):,.2f}`")
                else:
                    st.info(f"Received an unhandled event type: {event_type}", icon="⚠️")

    with col2:
        st.subheader("📈 Live Price Chart")
        if bids:
            chart_data = pd.DataFrame(bids, columns=['Time', 'Price']).set_index('Time')
            st.line_chart(chart_data)
        else:
            st.info("No bids have been placed yet.")

# --- Auto-refresh for live updates ---
time.sleep(1)
st.rerun() 
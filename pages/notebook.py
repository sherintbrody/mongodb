import streamlit as st
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import pytz
import calendar

st.set_page_config(page_title="Trading Notebook", layout="wide")

# --- Connect to MongoDB ---
@st.cache_resource
def init_connection():
    return MongoClient(st.secrets["mongo"]["URI"])

client = init_connection()
db = client[st.secrets["mongo"]["DB"]]
notebook_collection = db["notebook_entries"]

# Helper functions
def get_day_name(date_obj):
    return calendar.day_name[date_obj.weekday()]

def get_ist_timestamp():
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist).strftime("%d-%m-%Y %I:%M:%S %p")

# --- Main App ---
st.title("🗒️ Trading Notebook")
st.markdown("### Your personal trading diary")

# Simple entry form
st.subheader("✍️ Write Entry")

with st.form("notebook_form", clear_on_submit=True):
    # Date and Time
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        date_input = st.date_input("Date", datetime.today())
    
    with col2:
        day_name = get_day_name(date_input)
        st.text_input("Day", value=day_name, disabled=True)
    
    with col3:
        time_input = st.time_input("Time", datetime.now().time())
    
    # News field
    news = st.text_input("📰 News/Events (optional)", placeholder="Any important news or events...")
    
    # Main journal entry
    journal = st.text_area(
        "📝 Journal Entry",
        placeholder="Write your thoughts, observations, trades, lessons learned...",
        height=250
    )
    
    # Save button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.form_submit_button("💾 Save Entry", use_container_width=True, type="primary")
    
    if submitted:
        if journal or news:
            ist_timestamp = get_ist_timestamp()
            entry = {
                "date": date_input.strftime("%d-%m-%Y"),
                "day": day_name,
                "time": time_input.strftime("%I:%M %p"),
                "news": news,
                "journal": journal,
                "saved_at": ist_timestamp
            }
            
            notebook_collection.insert_one(entry)
            st.success(f"✅ Entry saved at {ist_timestamp} IST")
        else:
            st.error("Please write something in your journal")

st.divider()

# View entries
st.subheader("📖 Previous Entries")

# Simple filter
view_option = st.selectbox(
    "Show entries from",
    ["All", "Today", "Last 7 Days", "Last 30 Days"],
    index=0
)

# Build query based on filter
query = {}
if view_option == "Today":
    today = datetime.now().strftime("%d-%m-%Y")
    query["date"] = today

# Load entries (most recent first)
entries = list(notebook_collection.find(query).sort("_id", -1).limit(50))

if entries:
    for entry in entries:
        # Create a nice card-like display for each entry
        with st.container():
            # Header with date and time
            st.markdown(f"### 📅 {entry['date']} - {entry['day']} | {entry['time']}")
            
            # Display news if present
            if entry.get('news'):
                st.info(f"**📰 News:** {entry['news']}")
            
            # Display journal entry
            if entry.get('journal'):
                st.markdown(entry['journal'])
            
            # Show when it was saved
            st.caption(f"Saved at: {entry.get('saved_at', 'N/A')} IST")
            
            # Delete button
            col1, col2, col3 = st.columns([1, 4, 1])
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{entry['_id']}", type="secondary"):
                    notebook_collection.delete_one({"_id": entry['_id']})
                    st.success("Entry deleted")
                    st.rerun()
            
            st.divider()
else:
    st.info("📭 No entries yet. Start writing your trading diary!")

# Export option at the bottom
st.divider()
if st.button("📥 Export Diary to CSV"):
    all_entries = list(notebook_collection.find().sort("_id", -1))
    if all_entries:
        df = pd.DataFrame(all_entries)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"trading_diary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No entries to export")

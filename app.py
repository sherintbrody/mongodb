import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Trading Journal", layout="wide")

# --- Connect to MongoDB ---
uri = st.secrets["mongo"]["URI"]
db_name = st.secrets["mongo"]["DB"]
coll_name = st.secrets["mongo"]["COLLECTION"]

client = MongoClient(uri)
db = client[db_name]
collection = db[coll_name]

st.title("📒 Trading Journal Dashboard")

# --- Trade Entry Form ---
with st.form("trade_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol")
    with col2:
        side = st.selectbox("Side", ["BUY", "SELL"])
    with col3:
        qty = st.number_input("Quantity", min_value=1.0)
    price = st.number_input("Price", min_value=0.0)
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Save Trade")

    if submitted:
        collection.insert_one({
            "symbol": symbol.upper(),
            "side": side,
            "qty": qty,
            "price": price,
            "notes": notes
        })
        st.success("Trade saved ✅")

# --- Load Trades ---
docs = list(collection.find().sort("_id", -1))
if docs:
    df = pd.DataFrame(docs)
    df.drop(columns=["_id"], inplace=True)

    # --- Display Table ---
    st.subheader("📊 Trade History")
    st.dataframe(df, use_container_width=True)

    # --- P/L Analysis (if qty & price exist) ---
    if "qty" in df and "price" in df:
        df["value"] = df["qty"] * df["price"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", len(df))
            st.metric("Total Volume", df["qty"].sum())
        with col2:
            fig = px.histogram(df, x="symbol", color="side", barmode="group",
                               title="Trades by Symbol")
            st.plotly_chart(fig, use_container_width=True)

        # Timeline
        if "timestamp" in df:
            fig2 = px.line(df, x="timestamp", y="value", color="symbol",
                           title="Trade Value Over Time")
            st.plotly_chart(fig2, use_container_width=True)

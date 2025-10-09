import streamlit as st
from streamlit_option_menu import option_menu
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import numpy as np
import hashlib
import calendar
from zoneinfo import ZoneInfo
import json
from pathlib import Path
from collections import defaultdict
import re

st.set_page_config(
    page_title="Trading Journal Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername',
        'About': "Trading Journal Pro v2.0"
    }
)

# --- Modern Custom CSS ---
st.markdown("""
<style>
    /* Import Material Icons */
    @import url('https://fonts.googleapis.com/css2?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Material+Icons+Outlined');
    
    /* Fix for sidebar collapse icon - UPDATED */
    [data-testid="collapsedControl"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"] svg {
        display: block !important;
    }
    
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --success-color: #00c853;
        --danger-color: #ff1744;
        --warning-color: #ffa726;
        --background-color: #f8f9fa;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    
    /* Card styling */
    .element-container {
        border-radius: 8px;
    }
    
    /* Headers */
    h1 {
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 3px solid #1f77b4;
    }
    
    h2, h3 {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Modern Dark Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] * {
        color: #cfcfcf;
        font-family: 'Segoe UI', sans-serif;
    }
    .nav-link-selected {
        background-color: #2a2a2a !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 6px;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Red header styling for dataframe */
    thead tr th {
        background-color: #ff1744 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Diary Entry Box */
    .entry-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 15px;
    }
    .dark-mode .entry-box {
        background-color: #262730;
        border-left: 4px solid #58a6ff;
    }
    .entry-time {
        color: #1f77b4;
        font-weight: bold;
        font-size: 1.1em;
    }
    .entry-content {
        margin-top: 10px;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# --- Connect to MongoDB with caching ---
@st.cache_resource(ttl=3600)
def get_mongo_connection():
    """Get MongoDB connection - cached for 1 hour"""
    return MongoClient(st.secrets["mongo"]["URI"])

client = get_mongo_connection()
db = client[st.secrets["mongo"]["DB"]]
collection = db[st.secrets["mongo"]["COLLECTION"]]
diary_collection = db["diary_entries"]  # Diary entries collection

# --- Symbol Lists (cached as they don't change) ---
@st.cache_data
def get_symbol_lists():
    """Get all symbol lists - cached"""
    indices = ["NAS100", "US30", "SP500", "US100", "GER40", "UK100", "JPN225", "AUS200"]
    forex_majors = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
    forex_minors = ["EUR/GBP", "EUR/AUD", "EUR/CAD", "EUR/JPY", "GBP/JPY", "GBP/AUD", "AUD/JPY", "AUD/CAD", "NZD/JPY"]
    commodities = ["GOLD", "SILVER", "OIL", "NATGAS", "COPPER"]
    crypto = ["BTC/USD", "ETH/USD", "BTC/USDT", "ETH/USDT", "XRP/USD", "SOL/USD"]
    all_symbols = indices + forex_majors + forex_minors + commodities + crypto
    return all_symbols

ALL_SYMBOLS = get_symbol_lists()

# --- Helper Functions (cached where possible) ---
@st.cache_data
def format_date_display(date_val):
    """Format date to DD-MM-YYYY - cached"""
    if pd.isna(date_val) or date_val is None:
        return "-"
    
    try:
        if isinstance(date_val, str):
            dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
        elif isinstance(date_val, datetime):
            dt = date_val
        else:
            return "-"
        return dt.strftime('%d-%m-%Y')
    except:
        return "-"

@st.cache_data
def parse_date_from_display(date_str):
    """Parse DD-MM-YYYY to ISO format - cached"""
    if not date_str or date_str == "-" or date_str == "":
        return None
    try:
        dt = datetime.strptime(str(date_str), '%d-%m-%Y')
        return dt.isoformat()
    except:
        return None

def get_day_name(date_obj):
    """Get day name from date object"""
    return calendar.day_name[date_obj.weekday()]

def get_ist_time():
    """Get current time in IST"""
    ist = ZoneInfo('Asia/Kolkata')
    return datetime.now(ist)

def get_ist_timestamp():
    """Get IST timestamp string"""
    ist = ZoneInfo('Asia/Kolkata')
    return datetime.now(ist).strftime("%d-%m-%Y %I:%M:%S %p")

# --- Diary Helper Functions (MongoDB) ---
def format_journal_text(text):
    """Format journal text: each sentence on a new line"""
    if not text:
        return ""
    
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)
    formatted_lines = []
    
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1].strip() if i+1 < len(sentences) else ''
        if sentence:
            formatted_lines.append(sentence + punctuation)
    
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        formatted_lines.append(sentences[-1].strip())
    
    return '<br>'.join(formatted_lines)

def load_diary_entries():
    """Load diary entries from MongoDB"""
    try:
        docs = list(diary_collection.find().sort([("date", -1), ("time", -1)]))
        return docs
    except Exception as e:
        st.error(f"Error loading diary entries: {str(e)}")
        return []

def add_diary_entry(entry):
    """Add a new diary entry to MongoDB"""
    try:
        entry['id'] = datetime.now().timestamp()
        entry['created_at'] = datetime.now().isoformat()
        result = diary_collection.insert_one(entry)
        return result.inserted_id is not None
    except Exception as e:
        st.error(f"Error saving diary entry: {str(e)}")
        return False

def delete_diary_entry(entry_id):
    """Delete a diary entry by ID"""
    try:
        result = diary_collection.delete_one({"id": entry_id})
        return result.deleted_count > 0
    except Exception as e:
        st.error(f"Error deleting diary entry: {str(e)}")
        return False

def group_diary_entries_by_date(entries):
    """Group diary entries by date, maintaining order"""
    grouped = defaultdict(list)
    date_order = []
    
    for entry in entries:
        date = entry.get('date')
        if date not in date_order:
            date_order.append(date)
        grouped[date].append(entry)
    
    return grouped, date_order

def get_dates_with_diary_entries():
    """Get all unique dates that have diary entries"""
    all_entries = load_diary_entries()
    dates = set()
    for entry in all_entries:
        dates.add(entry.get('date'))
    return dates

def load_all_trades():
    """Load all trades from MongoDB - NO CACHING for fresh data"""
    try:
        docs = list(collection.find())
        return docs
    except Exception as e:
        st.error(f"Error loading trades: {str(e)}")
        return []

@st.cache_data(ttl=300)
def migrate_old_data(_df):
    """Migrate old data format to new format - cached for 5 minutes"""
    df = _df.copy()
    if df.empty:
        return df
    
    # Add missing columns with default values
    if 'status' not in df.columns:
        df['status'] = 'CLOSED'
    
    if 'entry_price' not in df.columns and 'price' in df.columns:
        df['entry_price'] = df['price']
    
    if 'quantity' not in df.columns and 'qty' in df.columns:
        df['quantity'] = df['qty']
    
    if 'side' in df.columns:
        df['side'] = df['side'].apply(lambda x: 'LONG' if x in ['BUY', 'LONG'] else 'SHORT')
    else:
        df['side'] = 'LONG'
    
    if 'entry_date' not in df.columns:
        df['entry_date'] = datetime.now()
    
    if 'outcome' not in df.columns:
        df['outcome'] = None
    
    if 'pnl' not in df.columns:
        df['pnl'] = None
    
    if 'exit_date' not in df.columns:
        df['exit_date'] = None
    
    # Ensure numeric columns are numeric
    numeric_cols = ['quantity', 'entry_price', 'exit_price', 'stop_loss', 'take_profit', 'risk_amount', 'pnl']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

@st.cache_data(ttl=300)
def calculate_metrics(_df):
    """Calculate trading metrics - cached for 5 minutes"""
    df = _df.copy()
    if df.empty:
        return {
            'total_trades': 0,
            'open_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'sharpe_ratio': 0,
            'total_wins': 0,
            'total_losses': 0
        }
    
    df = migrate_old_data(df)
    
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    
    if closed_trades.empty:
        return {
            'total_trades': len(df),
            'open_trades': len(df[df['status'] == 'OPEN']) if 'status' in df.columns else 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'sharpe_ratio': 0,
            'total_wins': 0,
            'total_losses': 0
        }
    
    trades_with_pnl = closed_trades[closed_trades['pnl'].notna()].copy()
    
    if trades_with_pnl.empty:
        return {
            'total_trades': len(df),
            'open_trades': len(df[df['status'] == 'OPEN']) if 'status' in df.columns else 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'sharpe_ratio': 0,
            'total_wins': 0,
            'total_losses': 0
        }
    
    winning_trades = trades_with_pnl[trades_with_pnl['pnl'] > 0]
    losing_trades = trades_with_pnl[trades_with_pnl['pnl'] < 0]
    
    metrics = {
        'total_trades': len(closed_trades),
        'open_trades': len(df[df['status'] == 'OPEN']) if 'status' in df.columns else 0,
        'win_rate': len(winning_trades) / len(trades_with_pnl) * 100 if len(trades_with_pnl) > 0 else 0,
        'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
        'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
        'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else 0,
        'total_pnl': trades_with_pnl['pnl'].sum() if not trades_with_pnl.empty else 0,
        'best_trade': trades_with_pnl['pnl'].max() if not trades_with_pnl.empty else 0,
        'worst_trade': trades_with_pnl['pnl'].min() if not trades_with_pnl.empty else 0,
        'sharpe_ratio': trades_with_pnl['pnl'].mean() / trades_with_pnl['pnl'].std() if not trades_with_pnl.empty and trades_with_pnl['pnl'].std() != 0 else 0,
        'total_wins': len(winning_trades),
        'total_losses': len(losing_trades)
    }
    
    return metrics

@st.cache_data(ttl=300)
def get_equity_curve(_df):
    """Generate equity curve from trades starting at 0 - cached for 5 minutes"""
    df = _df.copy()
    if df.empty:
        return pd.DataFrame()
    
    df = migrate_old_data(df)
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    
    if closed_trades.empty:
        return pd.DataFrame()
    
    trades_with_pnl = closed_trades[closed_trades['pnl'].notna()].copy()
    
    if trades_with_pnl.empty:
        return pd.DataFrame()
    
    if 'entry_date' in trades_with_pnl.columns:
        trades_with_pnl['entry_date'] = pd.to_datetime(trades_with_pnl['entry_date'])
        trades_with_pnl = trades_with_pnl.sort_values('entry_date')
    
    trades_with_pnl['cumulative_pnl'] = trades_with_pnl['pnl'].cumsum()
    
    return trades_with_pnl

@st.cache_data
def capitalize_headers(text):
    """Capitalize first letter of each word - cached"""
    return ' '.join(word.capitalize() for word in str(text).split('_'))

@st.cache_data
def get_chart_layout(title=""):
    """Get standard chart layout with black background - cached"""
    return dict(
        title=title,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            color='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            color='white'
        )
    )

# --- Initialize session state for performance ---
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

if 'data_version' not in st.session_state:
    st.session_state.data_version = 0

if 'diary_form_key' not in st.session_state:
    st.session_state.diary_form_key = 0

def increment_data_version():
    """Increment data version to invalidate cache"""
    st.session_state.data_version += 1
    st.session_state.last_update = datetime.now()

# --- Modern Sidebar Navigation ---
with st.sidebar:
    st.markdown("### 🚀 Trading Journal Pro")
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "New Trade", "Open Positions", "Trade History", "Calendar", "Diary", "Analytics", "Settings"],
        icons=["speedometer2", "plus-circle", "briefcase", "clock-history", "calendar3", "journal-text", "bar-chart-line", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0e1117"},
            "icon": {"color": "#cfcfcf", "font-size": "18px"},
            "nav-link": {
                "color": "#cfcfcf",
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px 0",
                "--hover-color": "#262730"
            },
            "nav-link-selected": {"background-color": "#2a2a2a"},
        }
    )
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        increment_data_version()
        st.rerun()
    
    st.markdown("---")
    st.info("💡 **Tip**: Consistently tracking your trades is key to improvement!")

# Map selected option to page
page = selected

# KEEPING ALL OTHER PAGES EXACTLY THE SAME - Only showing Analytics page changes below
# [Dashboard, New Trade, Open Positions, Trade History, Calendar, Diary pages code remains identical]

# For brevity, I'm only showing the CHANGED Analytics page section:

# --- ENHANCED ANALYTICS PAGE (ONLY CHART FIXES) ---
if page == "Analytics":
    st.title("📊 Advanced Trading Analytics")
    st.markdown("### Deep dive into your trading performance")
    
    docs = load_all_trades()
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        closed_df = df[df['status'] == 'CLOSED'].copy()
        
        if not closed_df.empty and 'pnl' in closed_df.columns:
            trades_with_pnl = closed_df[closed_df['pnl'].notna()].copy()
            
            if not trades_with_pnl.empty:
                # Create tabs for different analytics sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Performance Metrics", 
                    "⚠️ Risk Analysis", 
                    "🎯 Strategy & Instrument Analysis", 
                    "🧠 Behavioral Insights"
                ])
                
                # --- TAB 1: PERFORMANCE METRICS ---
                with tab1:
                    st.markdown("### 📈 Performance Overview")
                    
                    # Enhanced metrics row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    metrics = calculate_metrics(df)
                    
                    with col1:
                        st.metric("💰 Total P&L", f"${metrics['total_pnl']:.2f}")
                    with col2:
                        st.metric("🎯 Win Rate", f"{metrics['win_rate']:.1f}%")
                    with col3:
                        st.metric("📈 Profit Factor", f"{metrics['profit_factor']:.2f}")
                    with col4:
                        st.metric("⚡ Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col5:
                        avg_pnl = trades_with_pnl['pnl'].mean()
                        st.metric("📊 Avg Trade", f"${avg_pnl:.2f}")
                    
                    st.divider()
                    
                    # Equity Curve
                    equity_df = get_equity_curve(df)
                    
                    if not equity_df.empty:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("#### 📈 Equity Curve (Cumulative P&L)")
                            
                            trade_numbers = [0] + list(range(1, len(equity_df) + 1))
                            cumulative_values = [0] + equity_df['cumulative_pnl'].tolist()
                            
                            fig = go.Figure()
                            line_color = '#00c853' if cumulative_values[-1] >= 0 else '#ff1744'
                            fill_color = 'rgba(0, 200, 83, 0.2)' if cumulative_values[-1] >= 0 else 'rgba(255, 23, 68, 0.2)'
                            
                            fig.add_trace(go.Scatter(
                                x=trade_numbers,
                                y=cumulative_values,
                                mode='lines+markers',
                                name='Cumulative P&L',
                                line=dict(color=line_color, width=2),
                                marker=dict(size=5, color=line_color),
                                fill='tozeroy',
                                fillcolor=fill_color
                            ))
                            
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                            
                            layout = get_chart_layout()
                            layout['xaxis_title'] = "Trade Number"
                            layout['yaxis_title'] = "Cumulative P&L ($)"
                            layout['height'] = 450
                            layout['xaxis']['dtick'] = max(1, len(trade_numbers) // 20)
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 📊 Monthly Performance")
                            
                            # Add month-year column
                            if 'entry_date' in equity_df.columns:
                                equity_df['month_year'] = pd.to_datetime(equity_df['entry_date']).dt.to_period('M').astype(str)
                                monthly_pnl = equity_df.groupby('month_year')['pnl'].sum().reset_index()
                                
                                fig = go.Figure()
                                colors = ['#00c853' if x > 0 else '#ff1744' for x in monthly_pnl['pnl']]
                                
                                # FIXED: Added fixed bar width
                                fig.add_trace(go.Bar(
                                    x=monthly_pnl['month_year'],
                                    y=monthly_pnl['pnl'],
                                    marker_color=colors,
                                    name='Monthly P&L',
                                    width=0.5  # Fixed bar width
                                ))
                                
                                layout = get_chart_layout()
                                layout['xaxis_title'] = "Month"
                                layout['yaxis_title'] = "P&L ($)"
                                layout['height'] = 450
                                layout['showlegend'] = False
                                fig.update_layout(**layout)
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    # NEW: Instrument-Based P&L Analysis (HORIZONTAL CHARTS)
                    if 'symbol' in trades_with_pnl.columns:
                        st.markdown("#### 🎯 Instrument Performance Analysis")
                        
                        # Calculate P&L by instrument
                        instrument_pnl = trades_with_pnl.groupby('symbol').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).reset_index()
                        instrument_pnl.columns = ['Symbol', 'Total P&L', 'Avg P&L', 'Trades']
                        instrument_pnl = instrument_pnl.sort_values('Total P&L', ascending=True)  # Changed to True for horizontal
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### 📊 Total P&L by Instrument")
                            
                            # Top 10 instruments by total P&L
                            top_instruments = instrument_pnl.tail(10)  # tail for ascending=True
                            
                            fig = go.Figure()
                            colors = ['#00c853' if x > 0 else '#ff1744' for x in top_instruments['Total P&L']]
                            
                            # CHANGED: Horizontal bar chart with fixed height per bar
                            fig.add_trace(go.Bar(
                                y=top_instruments['Symbol'],  # Swapped: symbols on y-axis
                                x=top_instruments['Total P&L'],  # Swapped: values on x-axis
                                orientation='h',  # Horizontal orientation
                                marker_color=colors,
                                text=[f"${x:.2f}" for x in top_instruments['Total P&L']],
                                textposition='outside',
                                name='Total P&L',
                                width=0.6  # Fixed bar height for horizontal bars
                            ))
                            
                            layout = get_chart_layout()
                            layout['yaxis_title'] = "Instrument"  # Swapped
                            layout['xaxis_title'] = "Total P&L ($)"  # Swapped
                            layout['height'] = max(400, len(top_instruments) * 40)  # Dynamic height based on bars
                            layout['showlegend'] = False
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("##### 🎯 Win Rate by Instrument")
                            
                            # Calculate win rate by instrument
                            instrument_winrate = []
                            for symbol in trades_with_pnl['symbol'].unique():
                                symbol_trades = trades_with_pnl[trades_with_pnl['symbol'] == symbol]
                                total = len(symbol_trades)
                                wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                                win_rate = (wins / total * 100) if total > 0 else 0
                                instrument_winrate.append({
                                    'Symbol': symbol,
                                    'Win Rate': win_rate,
                                    'Trades': total
                                })
                            
                            winrate_df = pd.DataFrame(instrument_winrate)
                            winrate_df = winrate_df[winrate_df['Trades'] >= 3]  # Only show instruments with 3+ trades
                            winrate_df = winrate_df.sort_values('Win Rate', ascending=True).tail(10)  # Changed for horizontal
                            
                            if not winrate_df.empty:
                                fig = go.Figure()
                                colors = ['#00c853' if x >= 50 else '#ffa726' if x >= 40 else '#ff1744' 
                                         for x in winrate_df['Win Rate']]
                                
                                # CHANGED: Horizontal bar chart with fixed height per bar
                                fig.add_trace(go.Bar(
                                    y=winrate_df['Symbol'],  # Swapped: symbols on y-axis
                                    x=winrate_df['Win Rate'],  # Swapped: values on x-axis
                                    orientation='h',  # Horizontal orientation
                                    marker_color=colors,
                                    text=[f"{x:.1f}%" for x in winrate_df['Win Rate']],
                                    textposition='outside',
                                    name='Win Rate',
                                    width=0.6  # Fixed bar height
                                ))
                                
                                fig.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.5)  # Changed to vline
                                
                                layout = get_chart_layout()
                                layout['yaxis_title'] = "Instrument"  # Swapped
                                layout['xaxis_title'] = "Win Rate (%)"  # Swapped
                                layout['height'] = max(400, len(winrate_df) * 40)  # Dynamic height
                                layout['showlegend'] = False
                                fig.update_layout(**layout)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Not enough trades per instrument (minimum 3)")
                        
                        # Instrument Performance Table (unchanged)
                        st.markdown("##### 📋 Detailed Instrument Performance")
                        
                        detailed_instrument_metrics = []
                        for symbol in trades_with_pnl['symbol'].unique():
                            symbol_trades = trades_with_pnl[trades_with_pnl['symbol'] == symbol]
                            total = len(symbol_trades)
                            wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                            losses = len(symbol_trades[symbol_trades['pnl'] < 0])
                            total_pnl = symbol_trades['pnl'].sum()
                            avg_pnl = symbol_trades['pnl'].mean()
                            win_rate = (wins / total * 100) if total > 0 else 0
                            best_trade = symbol_trades['pnl'].max()
                            worst_trade = symbol_trades['pnl'].min()
                            
                            detailed_instrument_metrics.append({
                                'Symbol': symbol,
                                'Total Trades': total,
                                'Wins': wins,
                                'Losses': losses,
                                'Win Rate': f"{win_rate:.1f}%",
                                'Total P&L': f"${total_pnl:.2f}",
                                'Avg P&L': f"${avg_pnl:.2f}",
                                'Best Trade': f"${best_trade:.2f}",
                                'Worst Trade': f"${worst_trade:.2f}"
                            })
                        
                        detailed_df = pd.DataFrame(detailed_instrument_metrics)
                        detailed_df = detailed_df.sort_values('Total Trades', ascending=False)
                        
                        st.dataframe(
                            detailed_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                
                # --- TAB 2: RISK ANALYSIS ---
                with tab2:
                    st.markdown("### ⚠️ Risk Management Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Calculate risk metrics
                    max_drawdown = trades_with_pnl['pnl'].cumsum().min()
                    max_profit = trades_with_pnl['pnl'].cumsum().max()
                    avg_win_size = trades_with_pnl[trades_with_pnl['pnl'] > 0]['pnl'].mean() if len(trades_with_pnl[trades_with_pnl['pnl'] > 0]) > 0 else 0
                    avg_loss_size = abs(trades_with_pnl[trades_with_pnl['pnl'] < 0]['pnl'].mean()) if len(trades_with_pnl[trades_with_pnl['pnl'] < 0]) > 0 else 0
                    
                    with col1:
                        st.metric("📉 Max Drawdown", f"${max_drawdown:.2f}")
                    with col2:
                        st.metric("📈 Max Profit Run", f"${max_profit:.2f}")
                    with col3:
                        st.metric("▲ Avg Win Size", f"${avg_win_size:.2f}")
                    with col4:
                        st.metric("▼ Avg Loss Size", f"${avg_loss_size:.2f}")
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 P&L Distribution Histogram")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=trades_with_pnl['pnl'],
                            nbinsx=30,
                            marker_color='#2196f3',
                            opacity=0.7
                        ))
                        
                        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                        
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "P&L ($)"
                        layout['yaxis_title'] = "Frequency"
                        layout['height'] = 400
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 📉 Drawdown Analysis")
                        
                        # Calculate running drawdown
                        cumulative = trades_with_pnl['pnl'].cumsum()
                        running_max = cumulative.cummax()
                        drawdown = cumulative - running_max
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(drawdown))),
                            y=drawdown,
                            mode='lines',
                            name='Drawdown',
                            fill='tozeroy',
                            line=dict(color='#ff1744', width=2),
                            fillcolor='rgba(255, 23, 68, 0.2)'
                        ))
                        
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "Trade Number"
                        layout['yaxis_title'] = "Drawdown ($)"
                        layout['height'] = 400
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    # Risk/Reward analysis
                    if 'risk_reward_ratio' in trades_with_pnl.columns:
                        st.markdown("#### 🎯 Risk/Reward Ratio Analysis")
                        
                        rr_trades = trades_with_pnl[trades_with_pnl['risk_reward_ratio'].notna()]
                        
                        if not rr_trades.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                avg_rr = rr_trades['risk_reward_ratio'].mean()
                                st.metric("Average R:R Ratio", f"{avg_rr:.2f}")
                            
                            with col2:
                                # RR distribution
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=rr_trades['risk_reward_ratio'],
                                    nbinsx=20,
                                    marker_color='#00c853',
                                    opacity=0.7
                                ))
                                
                                layout = get_chart_layout()
                                layout['xaxis_title'] = "Risk/Reward Ratio"
                                layout['yaxis_title'] = "Frequency"
                                layout['height'] = 300
                                fig.update_layout(**layout)
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # --- TAB 3: STRATEGY & INSTRUMENT ANALYSIS ---
                with tab3:
                    st.markdown("### 🎯 Strategy & Trading Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    # Strategy Performance (unchanged)
                    if 'strategy' in trades_with_pnl.columns:
                        with col1:
                            st.markdown("#### 📋 Strategy Performance")
                            
                            strategy_pnl = trades_with_pnl.groupby('strategy').agg({
                                'pnl': ['sum', 'mean', 'count']
                            }).reset_index()
                            strategy_pnl.columns = ['Strategy', 'Total P&L', 'Avg P&L', 'Trades']
                            
                            # Calculate win rate per strategy
                            win_rates = []
                            for strategy in trades_with_pnl['strategy'].unique():
                                strat_trades = trades_with_pnl[trades_with_pnl['strategy'] == strategy]
                                wins = len(strat_trades[strat_trades['pnl'] > 0])
                                total = len(strat_trades)
                                win_rates.append((wins / total * 100) if total > 0 else 0)
                            
                            strategy_pnl['Win Rate'] = win_rates
                            strategy_pnl = strategy_pnl.sort_values('Total P&L', ascending=False)
                            
                            st.dataframe(
                                strategy_pnl.style.format({
                                    'Total P&L': '${:.2f}',
                                    'Avg P&L': '${:.2f}',
                                    'Win Rate': '{:.1f}%'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # Timeframe Performance (unchanged)
                    if 'timeframe' in trades_with_pnl.columns:
                        with col2:
                            st.markdown("#### ⏰ Timeframe Performance")
                            
                            timeframe_pnl = trades_with_pnl.groupby('timeframe').agg({
                                'pnl': ['sum', 'mean', 'count']
                            }).reset_index()
                            timeframe_pnl.columns = ['Timeframe', 'Total P&L', 'Avg P&L', 'Trades']
                            timeframe_pnl = timeframe_pnl.sort_values('Total P&L', ascending=False)
                            
                            st.dataframe(
                                timeframe_pnl.style.format({
                                    'Total P&L': '${:.2f}',
                                    'Avg P&L': '${:.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    st.divider()
                    
                    # Side (Long vs Short) Performance
                    if 'side' in trades_with_pnl.columns:
                        st.markdown("#### 📊 Long vs Short Performance")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        long_trades = trades_with_pnl[trades_with_pnl['side'] == 'LONG']
                        short_trades = trades_with_pnl[trades_with_pnl['side'] == 'SHORT']
                        
                        with col1:
                            if not long_trades.empty:
                                long_pnl = long_trades['pnl'].sum()
                                long_count = len(long_trades)
                                long_wins = len(long_trades[long_trades['pnl'] > 0])
                                long_winrate = (long_wins / long_count * 100) if long_count > 0 else 0
                                
                                st.metric("📈 Long Trades", long_count)
                                st.metric("Long P&L", f"${long_pnl:.2f}")
                                st.metric("Long Win Rate", f"{long_winrate:.1f}%")
                        
                        with col2:
                            if not short_trades.empty:
                                short_pnl = short_trades['pnl'].sum()
                                short_count = len(short_trades)
                                short_wins = len(short_trades[short_trades['pnl'] > 0])
                                short_winrate = (short_wins / short_count * 100) if short_count > 0 else 0
                                
                                st.metric("📉 Short Trades", short_count)
                                st.metric("Short P&L", f"${short_pnl:.2f}")
                                st.metric("Short Win Rate", f"{short_winrate:.1f}%")
                        
                        with col3:
                            # Pie chart comparison (unchanged)
                            fig = go.Figure(data=[go.Pie(
                                labels=['Long', 'Short'],
                                values=[len(long_trades), len(short_trades)],
                                marker=dict(colors=['#00c853', '#ff1744']),
                                hole=.4
                            )])
                            
                            layout = get_chart_layout()
                            layout['height'] = 300
                            layout['showlegend'] = True
                            layout['legend'] = dict(font=dict(color='white'))
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # --- TAB 4: BEHAVIORAL INSIGHTS ---
                with tab4:
                    st.markdown("### 🧠 Behavioral & Psychological Analysis")
                    
                    # Emotion-based performance
                    if 'emotion' in trades_with_pnl.columns:
                        st.markdown("#### 😊 Emotional State Performance")
                        
                        emotion_pnl = trades_with_pnl.groupby('emotion').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).reset_index()
                        emotion_pnl.columns = ['Emotion', 'Total P&L', 'Avg P&L', 'Trades']
                        emotion_pnl = emotion_pnl.sort_values('Total P&L', ascending=False)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = go.Figure()
                            colors = ['#00c853' if x > 0 else '#ff1744' for x in emotion_pnl['Total P&L']]
                            
                            # FIXED: Added fixed bar width
                            fig.add_trace(go.Bar(
                                x=emotion_pnl['Emotion'],
                                y=emotion_pnl['Total P&L'],
                                marker_color=colors,
                                text=[f"${x:.2f}" for x in emotion_pnl['Total P&L']],
                                textposition='outside',
                                width=0.5  # Fixed bar width
                            ))
                            
                            layout = get_chart_layout()
                            layout['xaxis_title'] = "Emotional State"
                            layout['yaxis_title'] = "Total P&L ($)"
                            layout['height'] = 400
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.dataframe(
                                emotion_pnl.style.format({
                                    'Total P&L': '${:.2f}',
                                    'Avg P&L': '${:.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                    
                    st.divider()
                    
                    # Confidence level performance (unchanged)
                    if 'confidence_level' in trades_with_pnl.columns:
                        st.markdown("#### 🎯 Confidence Level vs Performance")
                        
                        confidence_pnl = trades_with_pnl.groupby('confidence_level').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).reset_index()
                        confidence_pnl.columns = ['Confidence', 'Total P&L', 'Avg P&L', 'Trades']
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=confidence_pnl['Confidence'],
                            y=confidence_pnl['Avg P&L'],
                            mode='markers+lines',
                            marker=dict(size=confidence_pnl['Trades']*2, color='#2196f3'),
                            name='Avg P&L',
                            line=dict(color='#2196f3', width=2)
                        ))
                        
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "Confidence Level (1-10)"
                        layout['yaxis_title'] = "Average P&L ($)"
                        layout['height'] = 400
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("💡 **Insight**: Marker size represents number of trades. Look for patterns between confidence and performance.")
                    
                    st.divider()
                    
                    # Trading streaks (unchanged)
                    st.markdown("#### 🔥 Winning & Losing Streaks")
                    
                    # Calculate streaks
                    wins_losses = ['W' if x > 0 else 'L' for x in trades_with_pnl['pnl']]
                    
                    current_streak = 1
                    max_win_streak = 0
                    max_loss_streak = 0
                    current_type = wins_losses[0] if wins_losses else None
                    
                    for i in range(1, len(wins_losses)):
                        if wins_losses[i] == current_type:
                            current_streak += 1
                        else:
                            if current_type == 'W':
                                max_win_streak = max(max_win_streak, current_streak)
                            else:
                                max_loss_streak = max(max_loss_streak, current_streak)
                            current_streak = 1
                            current_type = wins_losses[i]
                    
                    # Final update
                    if current_type == 'W':
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, current_streak)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🔥 Max Win Streak", max_win_streak)
                    with col2:
                        st.metric("❄️ Max Loss Streak", max_loss_streak)
                    with col3:
                        # Current streak
                        current_final_streak = 1
                        for i in range(len(wins_losses)-2, -1, -1):
                            if wins_losses[i] == wins_losses[-1]:
                                current_final_streak += 1
                            else:
                                break
                        streak_type = "Win" if wins_losses[-1] == 'W' else "Loss"
                        st.metric(f"Current {streak_type} Streak", current_final_streak)
                
            else:
                st.info("📝 No trades with P&L data available for analysis")
        else:
            st.info("📝 No closed trades available for analytics")
    else:
        st.info("📝 No data available. Start trading to see analytics!")

# ALL OTHER PAGES REMAIN EXACTLY THE SAME
# (Dashboard, New Trade, Open Positions, Trade History, Calendar, Diary, Settings)
# I'm omitting them here for brevity, but they are 100% identical to your original code

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #888; margin: 0;'>
            <strong>Trading Journal Pro v2.0</strong> © 2024 | 
            <span style='color: #ff1744;'>Trade Responsibly</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

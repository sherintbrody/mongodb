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
    
    /* Fix for sidebar collapse icon */
    [data-testid="collapsedControl"] {
        font-family: 'Material Icons' !important;
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

# --- Dashboard Page ---
if page == "Dashboard":
    st.title("📊 Trading Dashboard")
    st.markdown("### Overview of Your Trading Performance")
    
    # Load fresh data from MongoDB
    docs = load_all_trades()
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        metrics = calculate_metrics(df)
        
        # Key Metrics Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            pnl_color = "normal" if metrics['total_pnl'] >= 0 else "inverse"
            st.metric(
                "💰 Total P&L",
                f"${metrics['total_pnl']:.2f}",
                delta=f"${metrics['total_pnl']:.2f}",
                delta_color=pnl_color
            )
        
        with col2:
            win_rate_delta = f"{metrics['win_rate']-50:.1f}%" if metrics['win_rate'] > 0 else None
            st.metric(
                "🎯 Win Rate",
                f"{metrics['win_rate']:.1f}%",
                delta=win_rate_delta
            )
        
        with col3:
            st.metric("📝 Total Trades", metrics['total_trades'])
        
        with col4:
            st.metric("📂 Open Positions", metrics['open_trades'])
        
        with col5:
            pf_delta = "Good ✓" if metrics['profit_factor'] > 1.5 else "Poor ✗" if metrics['profit_factor'] > 0 else None
            st.metric(
                "📈 Profit Factor",
                f"{metrics['profit_factor']:.2f}",
                delta=pf_delta
            )
        
        with col6:
            st.metric("🏆 Best Trade", f"${metrics['best_trade']:.2f}")
        
        st.divider()
        
        # Secondary Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Wins", metrics['total_wins'])
        
        with col2:
            st.metric("❌ Losses", metrics['total_losses'])
        
        with col3:
            st.metric("▲ Avg Win", f"${metrics['avg_win']:.2f}")
        
        with col4:
            st.metric("▼ Avg Loss", f"${metrics['avg_loss']:.2f}")
        
        st.divider()
        
        # Charts Row
        equity_df = get_equity_curve(df)
        
        if not equity_df.empty:
            st.markdown("### 📈 Equity Curve")
            
            # Equity Curve Line Chart with markers - Excel style
            fig = go.Figure()
            
            # Prepare data starting from 0
            trade_numbers = [0] + list(range(1, len(equity_df) + 1))
            cumulative_values = [0] + equity_df['cumulative_pnl'].tolist()
            
            # Determine color based on final value
            line_color = '#00c853' if cumulative_values[-1] >= 0 else '#ff1744'
            fill_color = 'rgba(0, 200, 83, 0.2)' if cumulative_values[-1] >= 0 else 'rgba(255, 23, 68, 0.2)'
            
            # Add line with markers
            fig.add_trace(go.Scatter(
                x=trade_numbers,
                y=cumulative_values,
                mode='lines+markers',
                name='Equity Curve',
                line=dict(
                    color=line_color,
                    width=2
                ),
                marker=dict(
                    size=6,
                    color=line_color,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                fill='tozeroy',
                fillcolor=fill_color
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            layout = get_chart_layout()
            layout['xaxis_title'] = "Trade Number"
            layout['yaxis_title'] = "Cumulative P&L ($)"
            layout['height'] = 500
            layout['hovermode'] = 'x unified'
            layout['showlegend'] = True
            layout['legend'] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='white')
            )
            layout['xaxis']['dtick'] = 1
            
            fig.update_layout(**layout)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Additional Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L Distribution
                st.markdown("### 📊 P&L Distribution")
                
                fig = go.Figure()
                
                trade_nums = list(range(1, len(equity_df) + 1))
                
                # Calculate fixed bar width based on number of trades
                bar_width = 0.4 if len(equity_df) > 50 else 0.6 if len(equity_df) > 20 else 0.8
                
                # Winning trades
                fig.add_trace(go.Bar(
                    x=trade_nums,
                    y=[pnl if pnl > 0 else 0 for pnl in equity_df['pnl']],
                    name='Wins',
                    marker_color='#00c853',
                    width=bar_width
                ))
                
                # Losing trades
                fig.add_trace(go.Bar(
                    x=trade_nums,
                    y=[pnl if pnl < 0 else 0 for pnl in equity_df['pnl']],
                    name='Losses',
                    marker_color='#ff1744',
                    width=bar_width
                ))
                
                layout = get_chart_layout()
                layout['xaxis_title'] = "Trade Number"
                layout['yaxis_title'] = "P&L ($)"
                layout['height'] = 400
                layout['barmode'] = 'relative'
                layout['showlegend'] = True
                layout['legend'] = dict(font=dict(color='white'))
                layout['xaxis']['dtick'] = 1
                
                fig.update_layout(**layout)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Trade Outcome Distribution
                if 'outcome' in df.columns:
                    st.markdown("### 🎯 Trade Outcomes")
                    outcome_counts = df[df['outcome'].notna()]['outcome'].value_counts()
                    
                    if not outcome_counts.empty:
                        colors = {
                            'WIN': '#00c853',
                            'LOSS': '#ff1744',
                            'BE': '#ffa726',
                            'TSL': '#2196f3'
                        }
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=outcome_counts.index,
                            values=outcome_counts.values,
                            hole=.4,
                            marker=dict(colors=[colors.get(x, '#808080') for x in outcome_counts.index])
                        )])
                        
                        layout = get_chart_layout()
                        layout['height'] = 400
                        layout['showlegend'] = True
                        layout['legend'] = dict(font=dict(color='white'))
                        
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No outcome data available")
                else:
                    # Symbol distribution as fallback
                    st.markdown("### 📊 Symbol Distribution")
                    if 'symbol' in df.columns:
                        symbol_counts = df['symbol'].value_counts().head(10)
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=symbol_counts.index,
                            values=symbol_counts.values,
                            hole=.4
                        )])
                        
                        layout = get_chart_layout()
                        layout['height'] = 400
                        layout['showlegend'] = True
                        layout['legend'] = dict(font=dict(color='white'))
                        
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 No closed trades with P&L data. Start trading and close positions to see your equity curve!")
        
        st.divider()
        
        # Recent Trades
        st.markdown("### 📋 Recent Trades")
        
        display_cols = []
        possible_cols = ['entry_date', 'exit_date', 'symbol', 'side', 'outcome', 'quantity', 'entry_price', 'exit_price', 'pnl', 'status']
        for col in possible_cols:
            if col in df.columns:
                display_cols.append(col)
        
        if display_cols:
            recent_trades = df.sort_index(ascending=False).head(10)[display_cols].copy()
            
            # Format dates
            if 'entry_date' in recent_trades.columns:
                recent_trades['entry_date'] = recent_trades['entry_date'].apply(format_date_display)
            
            if 'exit_date' in recent_trades.columns:
                recent_trades['exit_date'] = recent_trades['exit_date'].apply(format_date_display)
            
            # Format P&L
            if 'pnl' in recent_trades.columns:
                recent_trades['pnl'] = recent_trades['pnl'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "-"
                )
            
            st.dataframe(
                recent_trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "entry_date": st.column_config.TextColumn("Entry Date"),
                    "exit_date": st.column_config.TextColumn("Exit Date"),
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "side": st.column_config.TextColumn("Side"),
                    "outcome": st.column_config.TextColumn("Outcome"),
                    "pnl": st.column_config.TextColumn("P&L"),
                    "status": st.column_config.TextColumn("Status"),
                }
            )
    else:
        st.info("📝 No trades recorded yet. Start by adding a new trade!")

# --- New Trade Page ---
elif page == "New Trade":
    st.title("➕ Add New Trade")
    st.markdown("### Record your trade details")
    
    with st.form("advanced_trade_form", clear_on_submit=True):
        st.markdown("#### 📌 Basic Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Symbol dropdown with custom option
            symbol_choice = st.selectbox("Select Symbol*", ALL_SYMBOLS, index=0)
            if symbol_choice == "Custom":
                symbol = st.text_input("Enter Custom Symbol*", placeholder="e.g., AAPL, TSLA")
            else:
                symbol = symbol_choice
                st.text_input("Selected Symbol", value=symbol, disabled=True)
            
            side = st.selectbox("Side*", ["LONG", "SHORT"])
        
        with col2:
            trade_type = st.selectbox("Trade Type", ["FOREX", "INDICES", "COMMODITIES", "CRYPTO", "STOCK", "OPTIONS", "FUTURES"])
            status = st.selectbox("Status*", ["OPEN", "CLOSED"])
        
        with col3:
            entry_date = st.date_input("Entry Date*", value=date.today())
            # Close/Exit Date field next to Entry Date
            exit_date_input = st.date_input("Close Date", value=None if status == "OPEN" else date.today())
        
        col1, col2 = st.columns(2)
        with col1:
            entry_time = st.time_input("Entry Time*", value=datetime.now().time())
        with col2:
            exit_time = st.time_input("Close Time", value=datetime.now().time())
        
        st.divider()
        
        st.markdown("#### 💵 Position Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            quantity = st.number_input("Quantity*", min_value=0.001, step=0.01, value=1.0)
            entry_price = st.number_input("Entry Price*", min_value=0.0001, step=0.01)
        
        with col2:
            exit_price = st.number_input("Exit Price (if closed)", min_value=0.0, step=0.01)
            outcome = st.selectbox("Outcome*", ["PENDING", "WIN", "LOSS", "BE", "TSL"])
        
        with col3:
            entry_fee = st.number_input("Entry Fee ($)", min_value=0.0, step=0.01)
            exit_fee = st.number_input("Exit Fee ($)", min_value=0.0, step=0.01)
        
        st.divider()
        
        st.markdown("#### 🎯 Risk Management")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)
        with col2:
            take_profit = st.number_input("Take Profit", min_value=0.0, step=0.01)
        with col3:
            risk_amount = st.number_input("Risk Amount ($)", min_value=0.0, step=1.0)
        with col4:
            risk_reward_ratio = st.number_input("R:R Ratio", min_value=0.0, step=0.1)
        with col5:
            # P/L as a column in risk management
            pnl = st.number_input("P&L ($)*", step=0.01, format="%.2f", help="Enter your profit or loss")
        
        st.divider()
        
        st.markdown("#### 📝 Additional Information")
        
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["Day Trading", "Swing Trading", "Position Trading","Scalping", 
                 "Momentum", "Mean Reversion", "Breakout", "Other"]
            )
            timeframe = st.selectbox(
                "Timeframe",
                ["15m", "4H", "1m", "30m", "1H", "5m", "1D", "1W", "1M"]
            )
        
        with col2:
            confidence_level = st.slider("Confidence Level", 1, 10, 5)
            emotion = st.selectbox("Emotional State", ["Calm", "Excited", "Anxious", "Fearful", "Greedy"])
        
        notes = st.text_area("Trade Notes", placeholder="Entry reasons, market conditions, lessons learned...")
        
        tags = st.multiselect(
            "Tags",
            ["Earnings", "News", "Technical", "Fundamental", "FOMO", "Revenge Trade", "Plan Followed"]
        )
        
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            submitted = st.form_submit_button("💾 Save Trade", use_container_width=True, type="primary")
        
        if submitted:
            if symbol and quantity and entry_price:
                # Prepare exit_date
                exit_date_iso = None
                if exit_date_input and status == "CLOSED":
                    exit_date_iso = datetime.combine(exit_date_input, exit_time).isoformat()
                elif status == "CLOSED":
                    exit_date_iso = datetime.now().isoformat()
                
                trade_data = {
                    "symbol": symbol.upper(),
                    "side": side,
                    "trade_type": trade_type,
                    "quantity": float(quantity),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price) if exit_price > 0 else None,
                    "entry_date": datetime.combine(entry_date, entry_time).isoformat(),
                    "exit_date": exit_date_iso,
                    "status": status,
                    "outcome": outcome,
                    "pnl": float(pnl) if pnl != 0 else None,
                    "stop_loss": float(stop_loss) if stop_loss > 0 else None,
                    "take_profit": float(take_profit) if take_profit > 0 else None,
                    "risk_amount": float(risk_amount) if risk_amount > 0 else None,
                    "risk_reward_ratio": float(risk_reward_ratio) if risk_reward_ratio > 0 else None,
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "entry_fee": float(entry_fee),
                    "exit_fee": float(exit_fee),
                    "total_fees": float(entry_fee + exit_fee),
                    "notes": notes,
                    "tags": tags,
                    "confidence_level": int(confidence_level),
                    "emotion": emotion,
                    "created_at": datetime.now().isoformat()
                }
                
                result = collection.insert_one(trade_data)
                if result.inserted_id:
                    st.success("✅ Trade saved successfully!")
                    st.balloons()
                    # Clear cache and force refresh
                    st.cache_data.clear()
                    increment_data_version()
                    st.rerun()
                else:
                    st.error("❌ Failed to save trade")
            else:
                st.error("❌ Please fill in all required fields marked with *")

# --- Open Positions Page ---
elif page == "Open Positions":
    st.title("📈 Open Positions")
    st.markdown("### Active trades currently in the market")
    
    # Load fresh data
    docs = load_all_trades()
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        open_df = df[df['status'] == 'OPEN']
        
        if not open_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📂 Open Positions", len(open_df))
            with col2:
                total_value = (open_df['quantity'] * open_df['entry_price']).sum()
                st.metric("💰 Total Value", f"${total_value:.2f}")
            with col3:
                unique_symbols = open_df['symbol'].nunique()
                st.metric("📊 Unique Symbols", unique_symbols)
            with col4:
                long_positions = len(open_df[open_df['side'] == 'LONG'])
                st.metric("📈 Long Positions", long_positions)
            
            st.divider()
            
            # Display open positions with actions
            for idx, trade in open_df.iterrows():
                with st.expander(
                    f"**{trade['symbol']}** - {trade['side']} - Qty: {trade['quantity']} @ ${trade['entry_price']:.2f}",
                    expanded=False
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📅 Trade Information**")
                        st.write(f"Entry Date: {format_date_display(trade.get('entry_date', 'N/A'))}")
                        st.write(f"Strategy: {trade.get('strategy', 'N/A')}")
                        st.write(f"Timeframe: {trade.get('timeframe', 'N/A')}")
                        st.write(f"Type: {trade.get('trade_type', 'N/A')}")
                    
                    with col2:
                        st.markdown("**🎯 Risk Management**")
                        st.write(f"Stop Loss: ${trade.get('stop_loss', 'Not set')}")
                        st.write(f"Take Profit: ${trade.get('take_profit', 'Not set')}")
                        st.write(f"Risk Amount: ${trade.get('risk_amount', 'N/A')}")
                        st.write(f"R:R Ratio: {trade.get('risk_reward_ratio', 'N/A')}")
                    
                    with col3:
                        st.markdown("**⚡ Close Position**")
                        
                        # Close position form
                        with st.form(f"close_{trade['_id']}", clear_on_submit=True):
                            exit_price = st.number_input("Exit Price", min_value=0.01, step=0.01, key=f"exit_{trade['_id']}")
                            
                            outcome = st.selectbox(
                                "Outcome",
                                ["WIN", "LOSS", "BE", "TSL"],
                                key=f"outcome_{trade['_id']}"
                            )
                            
                            pnl_input = st.number_input(
                                "P&L ($)*",
                                step=0.01,
                                format="%.2f",
                                key=f"pnl_{trade['_id']}"
                            )
                            
                            exit_fee = st.number_input("Exit Fee", min_value=0.0, step=0.01, key=f"fee_{trade['_id']}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.form_submit_button("✅ Close", type="primary", use_container_width=True):
                                    update_data = {
                                        "exit_price": float(exit_price),
                                        "exit_fee": float(exit_fee),
                                        "exit_date": datetime.now().isoformat(),
                                        "status": "CLOSED",
                                        "outcome": outcome,
                                        "pnl": float(pnl_input)
                                    }
                                    
                                    result = collection.update_one(
                                        {"_id": trade['_id']},
                                        {"$set": update_data}
                                    )
                                    
                                    if result.modified_count > 0:
                                        st.success("✅ Position closed successfully!")
                                        st.cache_data.clear()
                                        increment_data_version()
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update position")
                            
                            with col_b:
                                if st.form_submit_button("🗑️ Delete", type="secondary", use_container_width=True):
                                    result = collection.delete_one({"_id": trade['_id']})
                                    if result.deleted_count > 0:
                                        st.success("🗑️ Trade deleted!")
                                        st.cache_data.clear()
                                        increment_data_version()
                                        st.rerun()
                    
                    if trade.get('notes'):
                        st.divider()
                        st.markdown("**📝 Notes:**")
                        st.info(trade['notes'])
        else:
            st.info("📭 No open positions")
    else:
        st.info("📝 No trades recorded yet")

# --- Trade History Page ---
elif page == "Trade History":
    st.title("📉 Trade History")
    st.markdown("### Complete record of all your trades")
    
    # Load ALL trades
    all_docs = list(collection.find())
    filtered_docs = all_docs
    
    if filtered_docs:
        # Create DataFrame from docs
        df = pd.DataFrame(filtered_docs)
        df = migrate_old_data(df)
        
        # Create a copy for MongoDB IDs
        df_ids = df['_id'].tolist()
        
        # Display summary
        col1, col2, col3, col4 = st.columns([1.5, 1, 1, 0.5])
        with col1:
            st.markdown(f"**Found {len(df)} trades**")
        with col2:
            if len(df) > 0 and 'pnl' in df.columns:
                total_filtered_pnl = df['pnl'].sum()
                color = "🟢" if total_filtered_pnl >= 0 else "🔴"
                st.markdown(f"**{color} P&L: ${total_filtered_pnl:.2f}**")
        with col3:
            if len(df) > 0 and 'pnl' in df.columns:
                wins = len(df[df['pnl'] > 0])
                losses = len(df[df['pnl'] < 0])
                st.markdown(f"**W: {wins} / L: {losses}**")
        with col4:
            pass  # Empty column for alignment
        
        st.divider()
        
        # Sort dataframe by entry_date descending (most recent first)
        if 'entry_date' in df.columns:
            df['entry_date_temp'] = pd.to_datetime(df['entry_date'], errors='coerce')
            df = df.sort_values('entry_date_temp', ascending=False)
            df = df.drop('entry_date_temp', axis=1)
            # Update df_ids after sorting
            df_ids = df['_id'].tolist()
        
        # Fixed display columns
        display_cols = ['entry_date', 'exit_date', 'symbol', 'side', 'outcome', 'quantity', 
                       'entry_price', 'exit_price', 'pnl', 'status', 'strategy', 'timeframe']
        
        # Filter to only include columns that exist
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Create editable dataframe
        edit_df = df[display_cols].copy()
        
        # Store original index mapping
        edit_df['original_index'] = df.reset_index(drop=True).index
        
        # Format for display
        for col in display_cols:
            if col in ['entry_date', 'exit_date']:
                edit_df[col] = edit_df[col].apply(format_date_display)
            elif edit_df[col].dtype == 'object':
                edit_df[col] = edit_df[col].fillna('').astype(str).replace('nan', '').replace('None', '')
            elif pd.api.types.is_numeric_dtype(edit_df[col]):
                edit_df[col] = edit_df[col].fillna(0)
        
        # Column configuration
        column_config = {}
        
        for col in display_cols:
            capitalized = capitalize_headers(col)
            
            if col in ['entry_date', 'exit_date']:
                column_config[col] = st.column_config.TextColumn(
                    capitalized,
                    help="Format: DD-MM-YYYY",
                    width="medium"
                )
            elif col in ['pnl', 'entry_price', 'exit_price', 'stop_loss', 'take_profit', 'risk_amount']:
                column_config[col] = st.column_config.NumberColumn(
                    capitalized,
                    format="%.2f",
                    width="medium"
                )
            elif col in ['quantity', 'risk_reward_ratio']:
                column_config[col] = st.column_config.NumberColumn(
                    capitalized,
                    format="%.2f",
                    width="medium"
                )
            else:
                column_config[col] = st.column_config.TextColumn(capitalized, width="medium")
        
        column_config['original_index'] = None  # Hide index column
        
        # Use dynamic key for data_editor
        editor_key = f"trade_editor_{len(df)}"
        
        # Editable data editor
        edited_df = st.data_editor(
            edit_df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            height=min(450, 50 + len(edit_df) * 35),  # Dynamic height based on rows
            key=editor_key
        )
        
        # Save changes button
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                try:
                    success_count = 0
                    # Update MongoDB with edited data
                    for idx in range(len(edited_df)):
                        original_idx = int(edited_df.iloc[idx]['original_index'])
                        trade_id = df_ids[original_idx]
                        
                        # Get edited row
                        row_data = edited_df.iloc[idx]
                        update_data = {}
                        
                        for col in display_cols:
                            new_value = row_data[col]
                            
                            # Skip if no change or empty
                            if pd.isna(new_value) or new_value == '' or new_value == '-':
                                continue
                            
                            # Handle dates
                            if col in ['entry_date', 'exit_date']:
                                iso_date = parse_date_from_display(str(new_value))
                                if iso_date:
                                    update_data[col] = iso_date
                            # Handle numeric fields
                            elif col in ['quantity', 'entry_price', 'exit_price', 'pnl', 'stop_loss', 
                                       'take_profit', 'risk_amount', 'risk_reward_ratio']:
                                try:
                                    update_data[col] = float(new_value)
                                except (ValueError, TypeError):
                                    pass
                            # Handle text fields
                            else:
                                update_data[col] = str(new_value)
                        
                        # Update in MongoDB
                        if update_data:
                            result = collection.update_one(
                                {"_id": trade_id},
                                {"$set": update_data}
                            )
                            if result.modified_count > 0:
                                success_count += 1
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully updated {success_count} trade(s)!")
                        st.cache_data.clear()
                        increment_data_version()
                        import time
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.info("ℹ️ No changes detected")
                        
                except Exception as e:
                    st.error(f"❌ Error saving changes: {str(e)}")
        
        st.divider()
        
        # Individual delete section
        st.markdown("### 🗑️ Delete Individual Trade")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create readable options for trade selection
            trade_options = []
            for idx, (_, row) in enumerate(df.iterrows()):
                entry_date = format_date_display(row.get('entry_date', 'N/A'))
                symbol = row.get('symbol', 'N/A')
                side = row.get('side', 'N/A')
                pnl = row.get('pnl', 0)
                pnl_str = f"${pnl:.2f}" if pd.notna(pnl) else "N/A"
                
                trade_options.append(f"{entry_date} | {symbol} | {side} | P&L: {pnl_str}")
            
            if trade_options:
                selected_trade_idx = st.selectbox(
                    "Select trade to delete",
                    range(len(trade_options)),
                    format_func=lambda x: trade_options[x]
                )
            else:
                selected_trade_idx = None
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete Selected", type="secondary", use_container_width=True):
                if selected_trade_idx is not None:
                    trade_id = df_ids[selected_trade_idx]
                    result = collection.delete_one({"_id": trade_id})
                    if result.deleted_count > 0:
                        st.success("✅ Trade deleted successfully!")
                        st.cache_data.clear()
                        increment_data_version()
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete trade")
        
        st.divider()
        
        # Export and bulk delete buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export DATA
            export_df = df.copy()
            if 'entry_date' in export_df.columns:
                export_df['entry_date'] = export_df['entry_date'].apply(format_date_display)
            if 'exit_date' in export_df.columns:
                export_df['exit_date'] = export_df['exit_date'].apply(format_date_display)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV ({len(df)} trades)",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            if st.button(f"🗑️ Delete All {len(df)} Trades", type="secondary", use_container_width=True):
                st.session_state['confirm_bulk_delete'] = True
        
        with col3:
            if st.session_state.get('confirm_bulk_delete', False):
                if st.button(f"⚠️ Confirm Delete {len(df)}", type="secondary", use_container_width=True):
                    result = collection.delete_many({})
                    st.success(f"✅ Deleted {result.deleted_count} trade(s)!")
                    st.session_state['confirm_bulk_delete'] = False
                    st.cache_data.clear()
                    increment_data_version()
                    st.rerun()
                
    else:
        st.info("📭 No trades found")

# --- Calendar Page ---
elif page == "Calendar":
    st.title("📅 Trading Calendar")
    st.markdown("### Visual overview of your trading activity")
    
    # Load trades
    docs = load_all_trades()
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        
        # Filter only closed trades with dates
        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
            df = df[df['entry_date'].notna()]
        
        if not df.empty:
            # Month/Year selector
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                selected_year = st.selectbox(
                    "Year",
                    options=sorted(df['entry_date'].dt.year.unique(), reverse=True),
                    index=0
                )
            
            with col2:
                month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                current_month = datetime.now().month
                selected_month_name = st.selectbox(
                    "Month",
                    options=month_names,
                    index=current_month - 1
                )
                selected_month = month_names.index(selected_month_name) + 1
            
            # Filter data for selected month
            month_df = df[(df['entry_date'].dt.year == selected_year) & 
                         (df['entry_date'].dt.month == selected_month)]
            
            # Calculate daily stats
            daily_stats = {}
            if not month_df.empty:
                for _, trade in month_df.iterrows():
                    day = trade['entry_date'].day
                    if day not in daily_stats:
                        daily_stats[day] = {
                            'trades': 0,
                            'wins': 0,
                            'losses': 0,
                            'pnl': 0,
                            'symbols': []
                        }
                    
                    daily_stats[day]['trades'] += 1
                    
                    if 'pnl' in trade and pd.notna(trade['pnl']):
                        daily_stats[day]['pnl'] += trade['pnl']
                        if trade['pnl'] > 0:
                            daily_stats[day]['wins'] += 1
                        elif trade['pnl'] < 0:
                            daily_stats[day]['losses'] += 1
                    
                    if 'symbol' in trade and pd.notna(trade['symbol']):
                        daily_stats[day]['symbols'].append(trade['symbol'])
            
            # Display calendar
            st.markdown("---")
            
            # Get calendar structure
            cal = calendar.monthcalendar(selected_year, selected_month)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create calendar header
            cols = st.columns(7)
            for idx, day in enumerate(days):
                with cols[idx]:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; color: #cfcfcf;'>{day[:3]}</div>", 
                               unsafe_allow_html=True)
            
            # Display calendar days
            for week in cal:
                cols = st.columns(7)
                for idx, day in enumerate(week):
                    with cols[idx]:
                        if day == 0:
                            st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
                        else:
                            # Check if we have data for this day
                            if day in daily_stats:
                                stats = daily_stats[day]
                                
                                # Determine background color based on P&L
                                if stats['pnl'] > 0:
                                    bg_color = "rgba(0, 200, 83, 0.2)"
                                    border_color = "#00c853"
                                    pnl_color = "#00c853"
                                elif stats['pnl'] < 0:
                                    bg_color = "rgba(255, 23, 68, 0.2)"
                                    border_color = "#ff1744"
                                    pnl_color = "#ff1744"
                                else:
                                    bg_color = "rgba(255, 167, 38, 0.2)"
                                    border_color = "#ffa726"
                                    pnl_color = "#ffa726"
                                
                                # Create day card
                                st.markdown(f"""
                                    <div style='
                                        background: {bg_color};
                                        border: 2px solid {border_color};
                                        border-radius: 8px;
                                        padding: 8px;
                                        height: 120px;
                                        margin: 2px;
                                    '>
                                        <div style='font-weight: bold; color: #cfcfcf;'>{day}</div>
                                        <div style='font-size: 0.8em; margin-top: 5px;'>
                                            <div>📊 {stats['trades']} trade(s)</div>
                                            <div style='color: #00c853;'>✅ {stats['wins']} win(s)</div>
                                            <div style='color: #ff1744;'>❌ {stats['losses']} loss(es)</div>
                                            <div style='color: {pnl_color}; font-weight: bold;'>
                                                P&L: ${stats['pnl']:.2f}
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Empty day (no trades)
                                st.markdown(f"""
                                    <div style='
                                        background: rgba(128, 128, 128, 0.1);
                                        border: 1px solid rgba(128, 128, 128, 0.3);
                                        border-radius: 8px;
                                        padding: 8px;
                                        height: 120px;
                                        margin: 2px;
                                    '>
                                        <div style='color: #808080;'>{day}</div>
                                        <div style='font-size: 0.8em; color: #606060; margin-top: 30px; text-align: center;'>
                                            No trades
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
            
            # Monthly Summary
            st.markdown("---")
            st.markdown("### 📊 Monthly Summary")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            # Calculate monthly stats
            month_total_trades = sum(stats['trades'] for stats in daily_stats.values())
            month_total_wins = sum(stats['wins'] for stats in daily_stats.values())
            month_total_losses = sum(stats['losses'] for stats in daily_stats.values())
            month_total_pnl = sum(stats['pnl'] for stats in daily_stats.values())
            month_win_rate = (month_total_wins / month_total_trades * 100) if month_total_trades > 0 else 0
            month_trading_days = len(daily_stats)
            
            with col1:
                st.metric("📅 Trading Days", month_trading_days)
            
            with col2:
                st.metric("📊 Total Trades", month_total_trades)
            
            with col3:
                st.metric("✅ Wins", month_total_wins)
            
            with col4:
                st.metric("❌ Losses", month_total_losses)
            
            with col5:
                st.metric("🎯 Win Rate", f"{month_win_rate:.1f}%")
            
            with col6:
                pnl_color = "normal" if month_total_pnl >= 0 else "inverse"
                st.metric("💰 Total P&L", f"${month_total_pnl:.2f}", delta_color=pnl_color)
            
            # Daily breakdown table
            st.markdown("---")
            st.markdown("### 📋 Daily Breakdown")
            
            if daily_stats:
                daily_data = []
                for day, stats in sorted(daily_stats.items()):
                    daily_data.append({
                        'Date': f"{selected_year}-{selected_month:02d}-{day:02d}",
                        'Trades': stats['trades'],
                        'Wins': stats['wins'],
                        'Losses': stats['losses'],
                        'P&L': f"${stats['pnl']:.2f}",
                        'Symbols': ', '.join(set(stats['symbols']))[:50]  # Limit symbol string length
                    })
                
                daily_df = pd.DataFrame(daily_data)
                st.dataframe(
                    daily_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "P&L": st.column_config.TextColumn("P&L"),
                        "Symbols": st.column_config.TextColumn("Symbols Traded")
                    }
                )
            
        else:
            st.info("📝 No trades with date information available")
    else:
        st.info("📝 No trades recorded yet")

# --- DIARY PAGE ---
elif page == "Diary":
    st.title("🗒️ Trading Diary")
    st.markdown("### Your personal trading journal and notes")
    
    # Simple entry form
    st.subheader("✍️ Write Entry")
    
    # Use separate form key for diary
    with st.form(f"diary_form_{st.session_state.diary_form_key}", clear_on_submit=True):
        # Date and Time
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            ist_now = get_ist_time()
            date_input = st.date_input("Date", ist_now.date())
        
        with col2:
            day_name = get_day_name(date_input)
            st.text_input("Day", value=day_name, disabled=True)
        
        with col3:
            current_ist_time = get_ist_time().time()
            time_input = st.time_input("Time (IST)", current_ist_time)
        
        # News field
        news = st.text_input("📰 News/Events ", placeholder="Any important news or events...")
        
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
                
                if add_diary_entry(entry):
                    st.success(f"✅ Entry saved at {ist_timestamp} IST")
                    st.session_state.diary_form_key += 1
                    st.rerun()
                else:
                    st.error("❌ Failed to save entry")
            else:
                st.error("Please write something in your journal")
    
    st.divider()
    
    # View entries
    st.subheader("📖 Previous Entries")
    
    # View options with tabs
    tab1, tab2 = st.tabs(["📅 Calendar View", "📋 List View"])
    
    # Load all diary entries
    all_entries = load_diary_entries()
    
    with tab1:
        st.markdown("#### Select a date to view entries")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            selected_date = st.date_input(
                "Pick a date",
                get_ist_time().date(),
                key="diary_calendar_picker"
            )
        
        with col2:
            dates_with_entries = get_dates_with_diary_entries()
            
        
        selected_date_str = selected_date.strftime("%d-%m-%Y")
        calendar_filtered = [e for e in all_entries if e.get('date') == selected_date_str]
        
        st.markdown("---")
        
        if calendar_filtered:
            day_name = get_day_name(selected_date)
            st.markdown(f"### 📅 {selected_date_str} - {day_name}")
            st.caption(f"💭 {len(calendar_filtered)} {'entry' if len(calendar_filtered) == 1 else 'entries'} on this day")
            
            for idx, entry in enumerate(calendar_filtered, 1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    with st.container():
                        st.markdown(f"""
                        <div class="entry-box">
                            <div class="entry-time">⏰ {entry['time']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if entry.get('news'):
                            st.info(f"**📰 News:** {entry['news']}")
                        
                        if entry.get('journal'):
                            formatted_journal = format_journal_text(entry['journal'])
                            st.markdown(f"<div class='entry-content'>{formatted_journal}</div>", unsafe_allow_html=True)
                        
                        
                
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Delete", key=f"diary_cal_delete_{entry.get('id')}", use_container_width=True):
                        if delete_diary_entry(entry.get('id')):
                            st.success("Entry deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete")
                
                if idx < len(calendar_filtered):
                    st.markdown("---")
        else:
            st.info(f"📭 No entries found for {selected_date_str}")
    
    with tab2:
        view_option = st.selectbox(
            "Show entries from",
            ["All", "Today", "Last 7 Days", "Last 30 Days"],
            index=0
        )
        
        filtered_entries = []
        if view_option == "All":
            filtered_entries = all_entries
        elif view_option == "Today":
            today = get_ist_time().strftime("%d-%m-%Y")
            filtered_entries = [e for e in all_entries if e.get('date') == today]
        elif view_option == "Last 7 Days":
            filtered_entries = all_entries[:20]
        elif view_option == "Last 30 Days":
            filtered_entries = all_entries[:50]
        
        st.markdown("---")
        
        if filtered_entries:
            grouped_entries, date_order = group_diary_entries_by_date(filtered_entries)
            
            for date in date_order:
                entries_for_date = grouped_entries[date]
                entry_count = len(entries_for_date)
                day_name = entries_for_date[0].get('day', '')
                
                st.markdown(f"### 📅 {date} - {day_name}")
                st.caption(f"💭 {entry_count} {'entry' if entry_count == 1 else 'entries'} on this day")
                
                for idx, entry in enumerate(entries_for_date):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        with st.container():
                            st.markdown(f"""
                            <div class="entry-box">
                                <div class="entry-time">⏰ {entry['time']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if entry.get('news'):
                                st.info(f"**📰 News:** {entry['news']}")
                            
                            if entry.get('journal'):
                                formatted_journal = format_journal_text(entry['journal'])
                                st.markdown(f"<div class='entry-content'>{formatted_journal}</div>", unsafe_allow_html=True)
                            
                            st.caption(f"📝 Saved at: {entry.get('saved_at', 'N/A')} IST")
                    
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("🗑️ Delete", key=f"diary_list_delete_{entry.get('id')}", use_container_width=True):
                            if delete_diary_entry(entry.get('id')):
                                st.success("Entry deleted")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                    
                    if idx < len(entries_for_date) - 1:
                        st.markdown("---")
                
                st.divider()
        else:
            st.info("📭 No entries yet. Start writing your trading diary!")
    
    # Export options
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Diary to CSV", use_container_width=True):
            all_entries = load_diary_entries()
            if all_entries:
                df = pd.DataFrame(all_entries)
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                if 'id' in df.columns:
                    df = df.drop('id', axis=1)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trading_diary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No entries to export")
    
    with col2:
        if st.button("📥 Export Diary to JSON", use_container_width=True):
            all_entries = load_diary_entries()
            if all_entries:
                export_entries = [{k: v for k, v in entry.items() if k not in ['_id', 'id']} for entry in all_entries]
                json_str = json.dumps(export_entries, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"trading_diary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            else:
                st.info("No entries to export")
    
    with st.expander("ℹ️ Diary Storage Info"):
        st.info(f"""
        Your diary entries are stored in MongoDB: `{db.name}.diary_entries`
        
        **Note:** 
        - Data is stored in MongoDB (cloud database)
        - All times are in IST (Indian Standard Time)
        - Each sentence in your journal will appear on a new line
        - Entries are grouped by date for easy viewing
        - Use Calendar View to see entries for specific dates
        - Use List View to browse all entries chronologically
        - You can export your data anytime using the export buttons above
        - Times are displayed in 12-hour format (AM/PM)
        - Data syncs across devices when using the same MongoDB database
        """)

# --- Analytics Page (keeping it short for character limit) ---
elif page == "Analytics":
    st.title("📊 Trading Analytics")
    st.markdown("### Deep dive into your trading performance")
    
    docs = load_all_trades()
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        closed_df = df[df['status'] == 'CLOSED'].copy()
        
        if not closed_df.empty and 'pnl' in closed_df.columns:
            trades_with_pnl = closed_df[closed_df['pnl'].notna()].copy()
            
            if not trades_with_pnl.empty:
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "⚠️ Risk Analysis", "🎯 Strategy Analysis", "🧠 Behavioral Analysis"])
                
                with tab1:
                    st.markdown("### Performance Metrics")
                    equity_df = get_equity_curve(df)
                    
                    if not equity_df.empty:
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
                            marker=dict(size=6, color=line_color, symbol='circle', line=dict(color='white', width=1)),
                            fill='tozeroy',
                            fillcolor=fill_color
                        ))
                        
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "Trade Number"
                        layout['yaxis_title'] = "Cumulative P&L ($)"
                        layout['height'] = 500
                        layout['xaxis']['dtick'] = 1
                        fig.update_layout(**layout)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.markdown("### Risk Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        max_drawdown = trades_with_pnl['pnl'].cumsum().min()
                        st.metric("📉 Max Drawdown", f"${max_drawdown:.2f}")
                    with col2:
                        max_profit = trades_with_pnl['pnl'].cumsum().max()
                        st.metric("📈 Max Profit", f"${max_profit:.2f}")
                
                with tab3:
                    if 'strategy' in trades_with_pnl.columns:
                        st.markdown("### Strategy Analysis")
                        st.info("Strategy data available")
                    else:
                        st.info("📝 No strategy data available")
                
                with tab4:
                    st.markdown("### Behavioral Analysis")
                    st.info("Behavioral data insights")
            else:
                st.info("📝 No trades with P&L data")
        else:
            st.info("📝 No closed trades available for analysis")
    else:
        st.info("📝 No data available for analytics")

# --- Settings Page ---
elif page == "Settings":
    st.title("⚙️ Settings")
    st.markdown("### Manage your trading journal")
    
    tab1, tab2, tab3 = st.tabs(["💾 Database", "🎨 Preferences", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Database Management")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = collection.count_documents({})
            st.metric("📊 Total Records", total_trades)
        
        with col2:
            open_trades = collection.count_documents({"status": "OPEN"})
            st.metric("📂 Open Trades", open_trades)
        
        with col3:
            closed_trades = collection.count_documents({"status": "CLOSED"})
            st.metric("✅ Closed Trades", closed_trades)
        
        with col4:
            diary_count = diary_collection.count_documents({})
            st.metric("📔 Diary Entries", diary_count)
    
    with tab2:
        st.markdown("### Preferences")
        st.info("Preferences settings")
    
    with tab3:
        st.markdown("### About Trading Journal Pro")
        st.markdown("""
        ## Trading Journal Pro v2.0
        
        **🚀 A comprehensive trading journal application**
        
        Features:
        - ✅ Track trades across multiple asset classes
        - 📊 Advanced analytics
        - 🗒️ Trading Diary with MongoDB storage
        - 📅 Calendar views
        """)

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

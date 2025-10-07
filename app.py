import streamlit as st
from streamlit_option_menu import option_menu
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import numpy as np
import hashlib

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

# --- Symbol Lists (cached as they don't change) ---
@st.cache_data
def get_symbol_lists():
    """Get all symbol lists - cached"""
    indices = ["NAS100", "US30", "SP500", "US100", "DJ30", "GER40", "UK100", "JPN225", "AUS200"]
    forex_majors = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
    forex_minors = ["EUR/GBP", "EUR/AUD", "EUR/CAD", "EUR/JPY", "GBP/JPY", "GBP/AUD", "AUD/JPY", "AUD/CAD", "NZD/JPY"]
    commodities = ["GOLD", "SILVER", "OIL", "NATGAS", "COPPER"]
    crypto = ["BTC/USD", "ETH/USD", "BTC/USDT", "ETH/USDT", "XRP/USD", "SOL/USD"]
    all_symbols = ["Custom"] + indices + forex_majors + forex_minors + commodities + crypto
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
    
    # Add outcome field if not present
    if 'outcome' not in df.columns:
        df['outcome'] = None
    
    # Add pnl field if not present
    if 'pnl' not in df.columns:
        df['pnl'] = None
    
    # Add exit_date if not present
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
    
    # Filter out trades without P&L data
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
    
    # Filter trades with P&L
    trades_with_pnl = closed_trades[closed_trades['pnl'].notna()].copy()
    
    if trades_with_pnl.empty:
        return pd.DataFrame()
    
    # Sort by entry date
    if 'entry_date' in trades_with_pnl.columns:
        trades_with_pnl['entry_date'] = pd.to_datetime(trades_with_pnl['entry_date'])
        trades_with_pnl = trades_with_pnl.sort_values('entry_date')
    
    # Calculate cumulative P&L
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
        options=["Dashboard", "New Trade", "Open Positions", "Trade History", "Analytics", "Settings"],
        icons=["speedometer2", "plus-circle", "briefcase", "clock-history", "bar-chart-line", "gear"],
        menu_icon=None,
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
    
    if st.button("🔄 Refresh Data", width="stretch"):
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
            st.metric("💚 Avg Win", f"${metrics['avg_win']:.2f}")
        
        with col4:
            st.metric("💔 Avg Loss", f"${metrics['avg_loss']:.2f}")
        
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
            
            st.plotly_chart(fig, width="stretch")
            
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
                
                st.plotly_chart(fig, width="stretch")
            
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
                        
                        st.plotly_chart(fig, width="stretch")
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
                        
                        st.plotly_chart(fig, width="stretch")
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
            quantity = st.number_input("Quantity*", min_value=0.0001, step=0.01, value=1.0)
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
                ["Scalping", "Day Trading", "Swing Trading", "Position Trading", 
                 "Momentum", "Mean Reversion", "Breakout", "Other"]
            )
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W", "1M"]
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
            submitted = st.form_submit_button("💾 Save Trade", width="stretch", type="primary")
        
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
                                if st.form_submit_button("✅ Close", type="primary", width="stretch"):
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
                                if st.form_submit_button("🗑️ Delete", type="secondary", width="stretch"):
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
    
    # Filters
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_symbol = st.text_input("Symbol", placeholder="All")
        with col2:
            filter_status = st.selectbox("Status", ["All", "OPEN", "CLOSED"])
        with col3:
            filter_side = st.selectbox("Side", ["All", "LONG", "SHORT"])
        with col4:
            filter_outcome = st.selectbox("Outcome", ["All", "WIN", "LOSS", "BE", "TSL", "PENDING"])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_strategy = st.selectbox(
                "Strategy",
                ["All", "Scalping", "Day Trading", "Swing Trading", "Position Trading", 
                 "Momentum", "Mean Reversion", "Breakout", "Other"]
            )
        with col2:
            filter_trade_type = st.selectbox(
                "Trade Type",
                ["All", "FOREX", "INDICES", "COMMODITIES", "CRYPTO", "STOCK", "OPTIONS", "FUTURES"]
            )
    
    # Build query
    query = {}
    if filter_symbol:
        query["symbol"] = {"$regex": filter_symbol.upper()}
    if filter_status != "All":
        query["status"] = filter_status
    if filter_side != "All":
        query["side"] = filter_side
    if filter_outcome != "All":
        query["outcome"] = filter_outcome
    if filter_strategy != "All":
        query["strategy"] = filter_strategy
    if filter_trade_type != "All":
        query["trade_type"] = filter_trade_type
    
    # Load fresh trades
    docs = list(collection.find(query))
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        
        # Create a copy for MongoDB IDs
        df_ids = df['_id'].tolist()
        
        # Display summary
        st.markdown(f"**Found {len(df)} trades**")
        
        st.divider()
        
        # Fixed display columns
        display_cols = ['entry_date', 'exit_date', 'symbol', 'side', 'outcome', 'quantity', 
                       'entry_price', 'exit_price', 'pnl', 'status', 'strategy', 'timeframe']
        
        # Filter to only include columns that exist
        display_cols = [col for col in display_cols if col in df.columns]
        
        # Create editable dataframe
        edit_df = df[display_cols].copy()
        
        # Store original index mapping
        edit_df['original_index'] = range(len(edit_df))
        
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
                    format="%.4f",
                    width="medium"
                )
            else:
                column_config[col] = st.column_config.TextColumn(capitalized, width="medium")
        
        column_config['original_index'] = None  # Hide index column
        
        # Editable data editor
        edited_df = st.data_editor(
            edit_df,
            column_config=column_config,
            hide_index=True,
            num_rows="fixed",
            height=450,
            key="trade_editor"
        )
        
        # Save changes button
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("💾 Save Changes", type="primary", width="stretch"):
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
                            # Handle numeric fields - IMPORTANT: Convert to float
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
                        # Clear cache and wait then refresh
                        st.cache_data.clear()
                        increment_data_version()
                        import time
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.info("ℹ️ No changes detected or all values were the same")
                        
                except Exception as e:
                    st.error(f"❌ Error saving changes: {str(e)}")
        
        st.divider()
        
        # Individual delete section
        st.markdown("### 🗑️ Delete Individual Trade")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create readable options for trade selection
            trade_options = []
            for idx, row in df.iterrows():
                entry_date = format_date_display(row.get('entry_date', 'N/A'))
                symbol = row.get('symbol', 'N/A')
                side = row.get('side', 'N/A')
                pnl = row.get('pnl', 0)
                pnl_str = f"${pnl:.2f}" if pd.notna(pnl) else "N/A"
                
                trade_options.append(f"{entry_date} | {symbol} | {side} | P&L: {pnl_str}")
            
            selected_trade_idx = st.selectbox(
                "Select trade to delete",
                range(len(trade_options)),
                format_func=lambda x: trade_options[x]
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete Selected", type="secondary", width="stretch"):
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
            # Export button
            export_df = df.copy()
            if 'entry_date' in export_df.columns:
                export_df['entry_date'] = export_df['entry_date'].apply(format_date_display)
            if 'exit_date' in export_df.columns:
                export_df['exit_date'] = export_df['exit_date'].apply(format_date_display)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
                type="primary"
            )
        
        with col2:
            if st.button("🗑️ Delete All Filtered", type="secondary", width="stretch"):
                st.session_state['confirm_bulk_delete'] = True
        
        with col3:
            if st.session_state.get('confirm_bulk_delete', False):
                if st.button("⚠️ Confirm Bulk Delete", type="secondary", width="stretch"):
                    result = collection.delete_many(query)
                    st.success(f"✅ Deleted {result.deleted_count} trade(s)!")
                    st.session_state['confirm_bulk_delete'] = False
                    st.cache_data.clear()
                    increment_data_version()
                    st.rerun()
                
    else:
        st.info("📭 No trades found with the selected filters")

# --- Analytics Page ---
elif page == "Analytics":
    st.title("📊 Trading Analytics")
    st.markdown("### Deep dive into your trading performance")
    
    # Load fresh data
    docs = load_all_trades()
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        
        # Only show analytics for closed trades with P&L data
        closed_df = df[df['status'] == 'CLOSED'].copy()
        
        if not closed_df.empty and 'pnl' in closed_df.columns:
            trades_with_pnl = closed_df[closed_df['pnl'].notna()].copy()
            
            if not trades_with_pnl.empty:
                # Tabs for different analytics
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "⚠️ Risk Analysis", "🎯 Strategy Analysis", "🧠 Behavioral Analysis"])
                
                with tab1:
                    st.markdown("### Performance Metrics")
                    
                    # Equity curve
                    equity_df = get_equity_curve(df)
                    
                    if not equity_df.empty:
                        st.markdown("#### 📈 Equity Curve (Cumulative P&L)")
                        
                        # Prepare data starting from 0
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
                        
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "Trade Number"
                        layout['yaxis_title'] = "Cumulative P&L ($)"
                        layout['height'] = 500
                        layout['hovermode'] = 'x unified'
                        layout['showlegend'] = True
                        layout['legend'] = dict(font=dict(color='white'))
                        layout['xaxis']['dtick'] = 1
                        
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, width="stretch")
                    
                    st.divider()
                    
                    # Performance by Symbol
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 💰 Total P&L by Symbol")
                        
                        symbol_performance = trades_with_pnl.groupby('symbol')['pnl'].sum().sort_values()
                        
                        fig = go.Figure()
                        
                        colors = ['#00c853' if x > 0 else '#ff1744' for x in symbol_performance.values]
                        
                        fig.add_trace(go.Bar(
                            y=symbol_performance.index,
                            x=symbol_performance.values,
                            orientation='h',
                            marker=dict(color=colors),
                            width=0.5
                        ))
                        
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "P&L ($)"
                        layout['yaxis_title'] = "Symbol"
                        layout['height'] = 400
                        layout['showlegend'] = False
                        
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, width="stretch")
                    
                    with col2:
                        st.markdown("#### 🎯 Win Rate by Symbol")
                        
                        win_rates = []
                        for symbol in trades_with_pnl['symbol'].unique():
                            symbol_trades = trades_with_pnl[trades_with_pnl['symbol'] == symbol]
                            wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                            total = len(symbol_trades)
                            win_rates.append({
                                'symbol': symbol,
                                'win_rate': (wins/total * 100) if total > 0 else 0,
                                'trades': total
                            })
                        
                        if win_rates:
                            win_rate_df = pd.DataFrame(win_rates).sort_values('win_rate')
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                y=win_rate_df['symbol'],
                                x=win_rate_df['win_rate'],
                                orientation='h',
                                marker=dict(color='#2196f3'),
                                text=win_rate_df['win_rate'].apply(lambda x: f'{x:.1f}%'),
                                textposition='outside',
                                width=0.5
                            ))
                            
                            layout = get_chart_layout()
                            layout['xaxis_title'] = "Win Rate (%)"
                            layout['yaxis_title'] = "Symbol"
                            layout['height'] = 400
                            layout['showlegend'] = False
                            layout['xaxis']['range'] = [0, 100]
                            
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, width="stretch")
                
                with tab2:
                    st.markdown("### Risk Analysis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        max_drawdown = trades_with_pnl['pnl'].cumsum().min()
                        st.metric("📉 Max Drawdown", f"${max_drawdown:.2f}")
                    
                    with col2:
                        max_profit = trades_with_pnl['pnl'].cumsum().max()
                        st.metric("📈 Max Profit", f"${max_profit:.2f}")
                    
                    with col3:
                        if 'risk_amount' in trades_with_pnl.columns:
                            avg_risk = trades_with_pnl['risk_amount'].mean()
                            st.metric("⚖️ Average Risk", f"${avg_risk:.2f}")
                    
                    with col4:
                        if 'risk_reward_ratio' in trades_with_pnl.columns:
                            avg_rr = trades_with_pnl['risk_reward_ratio'].mean()
                            st.metric("🎯 Avg R:R Ratio", f"{avg_rr:.2f}")
                    
                    st.divider()
                    
                    # Drawdown chart
                    st.markdown("#### 📉 Drawdown Chart")
                    
                    cumulative = trades_with_pnl['pnl'].cumsum()
                    running_max = cumulative.expanding().max()
                    drawdown = cumulative - running_max
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(drawdown) + 1)),
                        y=drawdown,
                        mode='lines',
                        name='Drawdown',
                        line=dict(color='#ff1744', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255, 23, 68, 0.3)'
                    ))
                    
                    layout = get_chart_layout()
                    layout['xaxis_title'] = "Trade Number"
                    layout['yaxis_title'] = "Drawdown ($)"
                    layout['height'] = 400
                    layout['showlegend'] = False
                    layout['xaxis']['dtick'] = 1
                    
                    fig.update_layout(**layout)
                    
                    st.plotly_chart(fig, width="stretch")
                    
                    st.divider()
                    
                    # Outcome distribution
                    if 'outcome' in trades_with_pnl.columns:
                        st.markdown("#### 🎯 Outcomes Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            outcome_counts = trades_with_pnl['outcome'].value_counts()
                            
                            colors_map = {
                                'WIN': '#00c853',
                                'LOSS': '#ff1744',
                                'BE': '#ffa726',
                                'TSL': '#2196f3'
                            }
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=outcome_counts.index,
                                values=outcome_counts.values,
                                marker=dict(colors=[colors_map.get(x, '#808080') for x in outcome_counts.index]),
                                hole=.4
                            )])
                            
                            layout = get_chart_layout("Trade Count by Outcome")
                            layout['height'] = 400
                            layout['showlegend'] = True
                            layout['legend'] = dict(font=dict(color='white'))
                            
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, width="stretch")
                        
                        with col2:
                            outcome_pnl = trades_with_pnl.groupby('outcome')['pnl'].sum()
                            
                            fig = go.Figure()
                            
                            colors = [colors_map.get(x, '#808080') for x in outcome_pnl.index]
                            
                            fig.add_trace(go.Bar(
                                x=outcome_pnl.index,
                                y=outcome_pnl.values,
                                marker=dict(color=colors),
                                width=0.5
                            ))
                            
                            layout = get_chart_layout("Total P&L by Outcome")
                            layout['xaxis_title'] = "Outcome"
                            layout['yaxis_title'] = "P&L ($)"
                            layout['height'] = 400
                            layout['showlegend'] = False
                            
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, width="stretch")
                
                with tab3:
                    if 'strategy' in trades_with_pnl.columns:
                        st.markdown("### Strategy Analysis")
                        
                        col1,

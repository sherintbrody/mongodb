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
        options=["Dashboard", "New Trade", "Open Positions", "Trade History", "Calendar", "📓 Daily Journal", "Analytics", "Settings"],
        icons=["speedometer2", "plus-circle", "briefcase", "clock-history", "calendar3", "bar-chart-line", "gear"],
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
    
    import calendar
    from datetime import datetime, date, timedelta
    
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

# --- Daily Journal Page ---
elif page == "📓 Daily Journal":
    st.title("📓 Daily Trading Journal")
    st.markdown("### Document your trading journey, thoughts, and lessons")
    
    # Create a separate collection for journal entries
    journal_collection = db["journal_entries"]
    
    # --- Journal Entry Form ---
    with st.container():
        st.subheader("✍️ New Journal Entry")
        
        with st.form("journal_form", clear_on_submit=True):
            # Date and Time Row
            col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
            with col1:
                entry_date = st.date_input("Date", datetime.today())
            with col2:
                day_of_week = entry_date.strftime("%A")
                st.text_input("Day", value=day_of_week, disabled=True)
            with col3:
                entry_time = st.time_input("Time", datetime.now().time())
            with col4:
                week_number = entry_date.isocalendar()[1]
                st.text_input("Week #", value=f"Week {week_number}", disabled=True)
            
            st.divider()
            
            # Market Conditions
            st.subheader("📈 Market Conditions")
            col1, col2, col3 = st.columns(3)
            with col1:
                market_trend = st.selectbox(
                    "Overall Market Trend",
                    ["", "Strong Uptrend", "Uptrend", "Ranging", "Downtrend", "Strong Downtrend"]
                )
            with col2:
                volatility = st.selectbox(
                    "Market Volatility",
                    ["", "Very Low", "Low", "Normal", "High", "Very High"]
                )
            with col3:
                market_sentiment = st.selectbox(
                    "Market Sentiment",
                    ["", "Extremely Bullish", "Bullish", "Neutral", "Bearish", "Extremely Bearish"]
                )
            
            # Important News/Events
            news_events = st.text_area(
                "📰 Important News/Events",
                placeholder="Fed announcement, earnings reports, economic data, etc.",
                height=80
            )
            
            st.divider()
            
            # Trading Psychology
            st.subheader("🧠 Trading Psychology")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                confidence_level = st.selectbox(
                    "Confidence Level",
                    ["", "Very Low", "Low", "Medium", "High", "Very High"]
                )
            with col2:
                emotional_state = st.selectbox(
                    "Emotional State",
                    ["", "Calm", "Focused", "Excited", "Anxious", "Fearful", "Greedy", "Frustrated"]
                )
            with col3:
                energy_level = st.slider("Energy Level", 1, 10, 5)
            with col4:
                focus_level = st.slider("Focus Level", 1, 10, 5)
            
            st.divider()
            
            # Trading Performance
            st.subheader("📊 Today's Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trades_taken = st.number_input("Trades Taken", min_value=0, step=1, value=0)
            with col2:
                winning_trades = st.number_input("Winning Trades", min_value=0, step=1, value=0)
            with col3:
                losing_trades = st.number_input("Losing Trades", min_value=0, step=1, value=0)
            with col4:
                daily_pnl = st.number_input("Daily P&L ($)", step=0.01, value=0.0)
            
            # Goals and Rules
            col1, col2 = st.columns(2)
            with col1:
                daily_goal = st.text_area(
                    "🎯 Daily Goal",
                    placeholder="What are you trying to achieve today?",
                    height=80
                )
            with col2:
                rules_followed = st.multiselect(
                    "✅ Rules Followed",
                    ["Risk Management", "Entry Rules", "Exit Rules", "Position Sizing", 
                     "Stop Loss", "No Revenge Trading", "No FOMO", "Stuck to Plan"]
                )
            
            st.divider()
            
            # Detailed Journal
            st.subheader("📝 Journal Entry")
            
            # Pre-market thoughts
            premarket_analysis = st.text_area(
                "Pre-Market Analysis",
                placeholder="What's your plan for today? Key levels, setups you're watching, etc.",
                height=100
            )
            
            # Trading notes
            trading_notes = st.text_area(
                "Trading Notes",
                placeholder="Document your trades, thought process, entries/exits, what worked/didn't work...",
                height=150
            )
            
            # Lessons learned
            lessons_learned = st.text_area(
                "Lessons Learned",
                placeholder="What did you learn today? What would you do differently?",
                height=100
            )
            
            # Tomorrow's plan
            tomorrow_plan = st.text_area(
                "Tomorrow's Plan",
                placeholder="What's the plan for tomorrow? Areas to improve, setups to watch...",
                height=80
            )
            
            # Rating
            col1, col2, col3 = st.columns(3)
            with col1:
                execution_rating = st.slider("Execution Rating", 1, 10, 5, help="How well did you execute your plan?")
            with col2:
                discipline_rating = st.slider("Discipline Rating", 1, 10, 5, help="How disciplined were you today?")
            with col3:
                overall_rating = st.slider("Overall Day Rating", 1, 10, 5, help="Overall satisfaction with today")
            
            # Tags
            tags = st.multiselect(
                "🏷️ Tags",
                ["Great Day", "Good Day", "Average Day", "Poor Day", "Lesson Day",
                 "Profitable", "Loss Day", "Break Even", "No Trades", "Overtraded",
                 "Followed Plan", "Broke Rules", "Emotional Trading", "Technical Issues"]
            )
            
            st.divider()
            
            # Submit button
            submitted = st.form_submit_button("💾 Save Journal Entry", use_container_width=True, type="primary")
            
            if submitted:
                journal_entry = {
                    "date": str(entry_date),
                    "day": day_of_week,
                    "time": str(entry_time),
                    "week_number": week_number,
                    "market_trend": market_trend,
                    "volatility": volatility,
                    "market_sentiment": market_sentiment,
                    "news_events": news_events,
                    "confidence_level": confidence_level,
                    "emotional_state": emotional_state,
                    "energy_level": energy_level,
                    "focus_level": focus_level,
                    "trades_taken": trades_taken,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "daily_pnl": daily_pnl,
                    "daily_goal": daily_goal,
                    "rules_followed": rules_followed,
                    "premarket_analysis": premarket_analysis,
                    "trading_notes": trading_notes,
                    "lessons_learned": lessons_learned,
                    "tomorrow_plan": tomorrow_plan,
                    "execution_rating": execution_rating,
                    "discipline_rating": discipline_rating,
                    "overall_rating": overall_rating,
                    "tags": tags,
                    "created_at": datetime.now().isoformat()
                }
                
                # Check if entry exists for this date
                existing_entry = journal_collection.find_one({"date": str(entry_date)})
                
                if existing_entry:
                    # Update existing entry
                    journal_collection.update_one(
                        {"date": str(entry_date)},
                        {"$set": journal_entry}
                    )
                    st.success(f"✅ Journal entry for {entry_date} updated successfully!")
                else:
                    # Insert new entry
                    journal_collection.insert_one(journal_entry)
                    st.success(f"✅ Journal entry for {entry_date} saved successfully!")
                
                st.balloons()
    
    st.divider()
    
    # --- View Journal Entries ---
    st.subheader("📚 Journal History")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        view_option = st.selectbox(
            "View",
            ["Last 7 Days", "Last 30 Days", "This Month", "All Entries", "Custom Range"]
        )
    
    with col2:
        if view_option == "Custom Range":
            start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
        else:
            start_date = None
    
    with col3:
        if view_option == "Custom Range":
            end_date = st.date_input("End Date", value=datetime.now().date())
        else:
            end_date = None
    
    # Build query based on filter
    query = {}
    if view_option == "Last 7 Days":
        start = (datetime.now() - timedelta(days=7)).date()
        query = {"date": {"$gte": str(start)}}
    elif view_option == "Last 30 Days":
        start = (datetime.now() - timedelta(days=30)).date()
        query = {"date": {"$gte": str(start)}}
    elif view_option == "This Month":
        start = datetime.now().replace(day=1).date()
        query = {"date": {"$gte": str(start)}}
    elif view_option == "Custom Range" and start_date and end_date:
        query = {"date": {"$gte": str(start_date), "$lte": str(end_date)}}
    
    # Load journal entries
    journal_docs = list(journal_collection.find(query).sort("date", -1))
    
    if journal_docs:
        # Display options
        display_mode = st.radio(
            "Display Mode",
            ["📋 Summary View", "📖 Detailed View", "📊 Table View"],
            horizontal=True
        )
        
        if display_mode == "📋 Summary View":
            # Summary cards for each entry
            for entry in journal_docs:
                with st.expander(f"📅 {entry['date']} - {entry['day']} | Rating: {entry.get('overall_rating', 'N/A')}/10"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Trades", entry.get('trades_taken', 0))
                        st.metric("Confidence", entry.get('confidence_level', 'N/A'))
                    
                    with col2:
                        win_rate = 0
                        if entry.get('trades_taken', 0) > 0:
                            win_rate = (entry.get('winning_trades', 0) / entry.get('trades_taken', 1)) * 100
                        st.metric("Win Rate", f"{win_rate:.0f}%")
                        st.metric("Emotion", entry.get('emotional_state', 'N/A'))
                    
                    with col3:
                        pnl = entry.get('daily_pnl', 0)
                        pnl_color = "🟢" if pnl >= 0 else "🔴"
                        st.metric("P&L", f"{pnl_color} ${pnl:.2f}")
                        st.metric("Discipline", f"{entry.get('discipline_rating', 0)}/10")
                    
                    with col4:
                        st.metric("Market", entry.get('market_trend', 'N/A'))
                        st.metric("Execution", f"{entry.get('execution_rating', 0)}/10")
                    
                    if entry.get('lessons_learned'):
                        st.markdown("**📚 Lessons Learned:**")
                        st.info(entry['lessons_learned'])
                    
                    if entry.get('tags'):
                        st.markdown("**🏷️ Tags:** " + ", ".join(entry['tags']))
        
        elif display_mode == "📖 Detailed View":
            # Full detailed view
            for entry in journal_docs:
                st.markdown(f"### 📅 {entry['date']} - {entry['day']}")
                
                # Performance metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Trades", entry.get('trades_taken', 0))
                with col2:
                    st.metric("Wins", entry.get('winning_trades', 0))
                with col3:
                    st.metric("Losses", entry.get('losing_trades', 0))
                with col4:
                    pnl = entry.get('daily_pnl', 0)
                    st.metric("P&L", f"${pnl:.2f}", delta=pnl)
                with col5:
                    st.metric("Overall Rating", f"{entry.get('overall_rating', 0)}/10")
                
                # Market conditions
                if entry.get('market_trend') or entry.get('news_events'):
                    st.markdown("**📈 Market Conditions:**")
                    if entry.get('market_trend'):
                        st.write(f"- Trend: {entry['market_trend']}")
                    if entry.get('volatility'):
                        st.write(f"- Volatility: {entry['volatility']}")
                    if entry.get('news_events'):
                        st.write(f"- News: {entry['news_events']}")
                
                # Journal content
                if entry.get('premarket_analysis'):
                    st.markdown("**Pre-Market Analysis:**")
                    st.text(entry['premarket_analysis'])
                
                if entry.get('trading_notes'):
                    st.markdown("**Trading Notes:**")
                    st.text(entry['trading_notes'])
                
                if entry.get('lessons_learned'):
                    st.markdown("**Lessons Learned:**")
                    st.success(entry['lessons_learned'])
                
                if entry.get('tomorrow_plan'):
                    st.markdown("**Tomorrow's Plan:**")
                    st.info(entry['tomorrow_plan'])
                
                st.divider()
        
        else:  # Table View
            # Create DataFrame
            df = pd.DataFrame(journal_docs)
            
            # Select columns to display
            display_columns = ['date', 'day', 'trades_taken', 'winning_trades', 
                             'losing_trades', 'daily_pnl', 'confidence_level', 
                             'emotional_state', 'overall_rating']
            
            # Filter to existing columns
            display_columns = [col for col in display_columns if col in df.columns]
            
            if display_columns:
                st.dataframe(
                    df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "daily_pnl": st.column_config.NumberColumn(
                            "Daily P&L",
                            format="$%.2f"
                        ),
                        "overall_rating": st.column_config.NumberColumn(
                            "Rating",
                            format="%d/10"
                        )
                    }
                )
            
            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Journal to CSV",
                data=csv,
                file_name=f"journal_entries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Statistics
        st.divider()
        st.subheader("📊 Journal Statistics")
        
        journal_df = pd.DataFrame(journal_docs)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_entries = len(journal_df)
            st.metric("Total Entries", total_entries)
        
        with col2:
            if 'daily_pnl' in journal_df.columns:
                total_pnl = journal_df['daily_pnl'].sum()
                avg_pnl = journal_df['daily_pnl'].mean()
                st.metric("Total P&L", f"${total_pnl:.2f}")
                st.metric("Avg Daily P&L", f"${avg_pnl:.2f}")
        
        with col3:
            if 'overall_rating' in journal_df.columns:
                avg_rating = journal_df['overall_rating'].mean()
                st.metric("Avg Rating", f"{avg_rating:.1f}/10")
        
        with col4:
            if 'trades_taken' in journal_df.columns:
                total_trades = journal_df['trades_taken'].sum()
                avg_trades = journal_df['trades_taken'].mean()
                st.metric("Total Trades", total_trades)
                st.metric("Avg Trades/Day", f"{avg_trades:.1f}")
        
        # Emotional state analysis
        if 'emotional_state' in journal_df.columns:
            emotion_counts = journal_df['emotional_state'].value_counts()
            if not emotion_counts.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotional States Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # P&L by emotion
                    if 'daily_pnl' in journal_df.columns:
                        emotion_pnl = journal_df.groupby('emotional_state')['daily_pnl'].mean()
                        fig = px.bar(
                            x=emotion_pnl.index,
                            y=emotion_pnl.values,
                            title="Average P&L by Emotional State",
                            labels={'x': 'Emotional State', 'y': 'Avg P&L ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 No journal entries found for the selected period")
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if 'strategy' in trades_with_pnl.columns:
                        st.markdown("### Strategy Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Trades by Strategy")
                            
                            strategy_counts = trades_with_pnl['strategy'].value_counts()
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=strategy_counts.index,
                                values=strategy_counts.values,
                                hole=.4
                            )])
                            
                            layout = get_chart_layout()
                            layout['height'] = 400
                            layout['showlegend'] = True
                            layout['legend'] = dict(font=dict(color='white'))
                            
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 💰 P&L by Strategy")
                            
                            strategy_pnl = trades_with_pnl.groupby('strategy')['pnl'].sum().sort_values()
                            
                            fig = go.Figure()
                            
                            colors = ['#00c853' if x > 0 else '#ff1744' for x in strategy_pnl.values]
                            
                            fig.add_trace(go.Bar(
                                y=strategy_pnl.index,
                                x=strategy_pnl.values,
                                orientation='h',
                                marker=dict(color=colors),
                                width=0.5
                            ))
                            
                            layout = get_chart_layout()
                            layout['xaxis_title'] = "Total P&L ($)"
                            layout['yaxis_title'] = "Strategy"
                            layout['height'] = 400
                            layout['showlegend'] = False
                            
                            fig.update_layout(**layout)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        # Strategy performance table
                        st.markdown("#### 📋 Strategy Performance Table")
                        
                        strategy_stats = []
                        for strategy in trades_with_pnl['strategy'].unique():
                            strategy_trades = trades_with_pnl[trades_with_pnl['strategy'] == strategy]
                            wins = len(strategy_trades[strategy_trades['pnl'] > 0])
                            total = len(strategy_trades)
                            
                            strategy_stats.append({
                                'Strategy': strategy,
                                'Total Trades': total,
                                'Wins': wins,
                                'Losses': total - wins,
                                'Win Rate (%)': f"{(wins/total * 100):.1f}" if total > 0 else "0",
                                'Total P&L ($)': f"{strategy_trades['pnl'].sum():.2f}",
                                'Avg P&L ($)': f"{strategy_trades['pnl'].mean():.2f}"
                            })
                        
                        strategy_df = pd.DataFrame(strategy_stats)
                        st.dataframe(strategy_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("📝 No strategy data available")
                
                with tab4:
                    st.markdown("### Behavioral Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'emotion' in trades_with_pnl.columns:
                            st.markdown("#### 😊 P&L by Emotional State")
                            
                            emotion_performance = trades_with_pnl.groupby('emotion')['pnl'].mean().sort_values()
                            
                            if not emotion_performance.empty:
                                fig = go.Figure()
                                
                                colors = ['#00c853' if x > 0 else '#ff1744' for x in emotion_performance.values]
                                
                                fig.add_trace(go.Bar(
                                    y=emotion_performance.index,
                                    x=emotion_performance.values,
                                    orientation='h',
                                    marker=dict(color=colors),
                                    width=0.5
                                ))
                                
                                layout = get_chart_layout()
                                layout['xaxis_title'] = "Average P&L ($)"
                                layout['yaxis_title'] = "Emotional State"
                                layout['height'] = 400
                                layout['showlegend'] = False
                                
                                fig.update_layout(**layout)
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'confidence_level' in trades_with_pnl.columns:
                            st.markdown("#### 🎯 Confidence vs P&L")
                            
                            confidence_data = trades_with_pnl[['confidence_level', 'pnl']].dropna()
                            
                            if not confidence_data.empty:
                                fig = go.Figure()
                                
                                colors = ['#00c853' if x > 0 else '#ff1744' for x in confidence_data['pnl']]
                                
                                fig.add_trace(go.Scatter(
                                    x=confidence_data['confidence_level'],
                                    y=confidence_data['pnl'],
                                    mode='markers',
                                    marker=dict(
                                        size=10,
                                        color=colors,
                                        line=dict(width=1, color='white')
                                    )
                                ))
                                
                                layout = get_chart_layout()
                                layout['xaxis_title'] = "Confidence Level"
                                layout['yaxis_title'] = "P&L ($)"
                                layout['height'] = 400
                                layout['showlegend'] = False
                                
                                fig.update_layout(**layout)
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    # Time-based analysis
                    if 'entry_date' in trades_with_pnl.columns:
                        st.markdown("#### 📅 Performance Over Time")
                        
                        # Convert entry_date to datetime
                        trades_with_pnl['entry_date'] = pd.to_datetime(trades_with_pnl['entry_date'])
                        
                        # Group by day
                        daily_pnl = trades_with_pnl.groupby(trades_with_pnl['entry_date'].dt.date)['pnl'].sum()
                        
                        fig = go.Figure()
                        
                        colors = ['#00c853' if x > 0 else '#ff1744' for x in daily_pnl.values]
                        
                        # Calculate bar width based on number of days
                        bar_width = 0.3 if len(daily_pnl) > 30 else 0.6
                        
                        fig.add_trace(go.Bar(
                            x=daily_pnl.index,
                            y=daily_pnl.values,
                            marker=dict(color=colors),
                            width=bar_width
                        ))
                        
                        layout = get_chart_layout()
                        layout['xaxis_title'] = "Date"
                        layout['yaxis_title'] = "Daily P&L ($)"
                        layout['height'] = 400
                        layout['showlegend'] = False
                        
                        fig.update_layout(**layout)
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📝 No trades with P&L data. Add P&L to trades to see analytics.")
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
        
        # Database stats
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
            unique_symbols = len(collection.distinct("symbol"))
            st.metric("📈 Unique Symbols", unique_symbols)
        
        st.divider()
        
        # Backup
        st.markdown("### 📥 Backup & Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Backup Database", use_container_width=True, type="primary"):
                all_trades = load_all_trades()
                if all_trades:
                    df = pd.DataFrame(all_trades)
                    df['_id'] = df['_id'].astype(str)
                    
                    # Format dates for export
                    if 'entry_date' in df.columns:
                        df['entry_date'] = df['entry_date'].apply(format_date_display)
                    if 'exit_date' in df.columns:
                        df['exit_date'] = df['exit_date'].apply(format_date_display)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Backup",
                        data=csv,
                        file_name=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No data to backup")
        
        st.divider()
        
        # Danger Zone
        st.markdown("### ⚠️ Danger Zone")
        
        with st.expander("🗑️ Delete Operations", expanded=False):
            st.warning("⚠️ These actions cannot be undone!")
            
            st.markdown("#### Delete All Trades")
            if st.checkbox("✅ I understand this will delete ALL trades permanently"):
                if st.button("🗑️ Confirm Delete All Trades", type="secondary"):
                    collection.delete_many({})
                    st.success("✅ All trades deleted")
                    st.cache_data.clear()
                    increment_data_version()
                    st.rerun()
            
            st.divider()
            
            st.markdown("#### Delete All Open Trades")
            if st.checkbox("✅ I understand this will delete all OPEN trades"):
                if st.button("🗑️ Confirm Delete Open Trades", type="secondary"):
                    collection.delete_many({"status": "OPEN"})
                    st.success("✅ Open trades deleted")
                    st.cache_data.clear()
                    increment_data_version()
                    st.rerun()
            
            st.divider()
            
            st.markdown("#### Delete All Closed Trades")
            if st.checkbox("✅ I understand this will delete all CLOSED trades"):
                if st.button("🗑️ Confirm Delete Closed Trades", type="secondary"):
                    collection.delete_many({"status": "CLOSED"})
                    st.success("✅ Closed trades deleted")
                    st.cache_data.clear()
                    increment_data_version()
                    st.rerun()
    
    with tab2:
        st.markdown("### Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💱 Display Settings")
            currency = st.selectbox("Default Currency", ["USD", "EUR", "GBP", "JPY", "BTC", "ETH"])
            date_format = st.selectbox("Date Format", ["DD-MM-YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
            decimal_places = st.number_input("Decimal Places", min_value=0, max_value=8, value=2)
        
        with col2:
            st.markdown("#### 📊 Chart Settings")
            chart_theme = st.selectbox("Chart Theme", ["Dark", "Light"])
            default_chart_height = st.number_input("Chart Height (px)", min_value=200, max_value=1000, value=400, step=50)
            show_grid = st.checkbox("Show Grid Lines", value=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("💾 Save Preferences", use_container_width=True, type="primary"):
                st.success("✅ Preferences saved!")
    
    with tab3:
        st.markdown("### About Trading Journal Pro")
        
        st.markdown("""
        ## Trading Journal Pro v2.0
        
        **🚀 A comprehensive trading journal application**
        
        Built with modern technologies to help traders track, analyze, and improve their trading performance.
        
        **Version:** 2.0.0  
        **Last Updated:** 2024
        """)
        
        st.divider()
        
        st.markdown("### ✨ Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - ✅ Track trades across multiple asset classes
            - 📊 Advanced analytics and performance metrics
            - 📈 Equity curve visualization with markers
            - 🎯 Win/Loss/TSL/BE outcome tracking
            - 💰 Manual P&L entry
            - ⚠️ Risk management tools
            """)
        
        with col2:
            st.markdown("""
            - 🎯 Strategy analysis
            - 🧠 Behavioral tracking
            - 📥 Data export and backup
            - 🗑️ Flexible delete options
            - ✏️ Editable trade history
            - 🔄 Real-time data sync
            """)
        
        st.divider()
        
        st.markdown("### 🛠️ Tech Stack")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("**Frontend**\n\nStreamlit")
        
        with col2:
            st.info("**Database**\n\nMongoDB")
        
        with col3:
            st.info("**Charts**\n\nPlotly")
        
        with col4:
            st.info("**Data**\n\nPandas")
        
        st.divider()
        
        st.markdown("### 💡 Tips for Success")
        st.info("""
        **1. Consistency is Key** - Log every trade, no matter how small  
        **2. Be Honest** - Record your emotions and mistakes  
        **3. Review Regularly** - Analyze your stats weekly  
        **4. Set Goals** - Use data to improve your win rate  
        **5. Learn from Losses** - They're your best teachers
        """)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #888; margin: 0;'>
            <strong>Trading Journal Pro v2.0</strong> © 2024 | 
            <span style='color: #ff1744;'>Trade Responsibly</span>
        </p>
        <p style='color: #aaa; font-size: 0.9em; margin-top: 10px;'>
            💡 Remember: Past performance does not guarantee future results
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

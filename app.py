import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import numpy as np

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
    
    /* Sidebar - Keep default Streamlit styling */
    [data-testid="stSidebar"] {
        background-color: inherit;
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
    
    /* Delete button styling */
    .delete-button {
        background-color: #ff1744;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# --- Connect to MongoDB ---
@st.cache_resource
def init_connection():
    return MongoClient(st.secrets["mongo"]["URI"])

client = init_connection()
db = client[st.secrets["mongo"]["DB"]]
collection = db[st.secrets["mongo"]["COLLECTION"]]

# --- Symbol Lists ---
INDICES = ["NAS100", "US30", "SP500", "US100", "DJ30", "GER40", "UK100", "JPN225", "AUS200"]

FOREX_MAJORS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"
]

FOREX_MINORS = [
    "EUR/GBP", "EUR/AUD", "EUR/CAD", "EUR/JPY", "GBP/JPY", "GBP/AUD", 
    "AUD/JPY", "AUD/CAD", "NZD/JPY"
]

COMMODITIES = ["GOLD", "SILVER", "OIL", "NATGAS", "COPPER"]

CRYPTO = ["BTC/USD", "ETH/USD", "BTC/USDT", "ETH/USDT", "XRP/USD", "SOL/USD"]

ALL_SYMBOLS = ["Custom"] + INDICES + FOREX_MAJORS + FOREX_MINORS + COMMODITIES + CRYPTO

# --- Helper Functions ---
def migrate_old_data(df):
    """Migrate old data format to new format"""
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
    
    # Ensure numeric columns are numeric
    numeric_cols = ['quantity', 'entry_price', 'exit_price', 'stop_loss', 'take_profit', 'risk_amount', 'pnl']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_metrics(df):
    """Calculate trading metrics"""
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

def get_equity_curve(df):
    """Generate equity curve from trades"""
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

# --- Sidebar Navigation ---
st.sidebar.title("🚀 Trading Journal Pro")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📋 Navigation",
    ["📊 Dashboard", "➕ New Trade", "📈 Open Positions", "📉 Trade History", 
     "📊 Analytics", "⚙️ Settings"],
    label_visibility="visible"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Consistently tracking your trades is key to improvement!")

# --- Dashboard Page ---
if page == "📊 Dashboard":
    st.title("📊 Trading Dashboard")
    st.markdown("### Overview of Your Trading Performance")
    
    # Load all trades
    docs = list(collection.find())
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
            
            # Equity Curve Line Chart
            fig = go.Figure()
            
            # Add cumulative P&L line
            fig.add_trace(go.Scatter(
                x=list(range(1, len(equity_df) + 1)),
                y=equity_df['cumulative_pnl'],
                mode='lines',
                name='Equity Curve',
                line=dict(
                    color='#00c853' if equity_df['cumulative_pnl'].iloc[-1] > 0 else '#ff1744',
                    width=3
                ),
                fill='tozeroy',
                fillcolor='rgba(0, 200, 83, 0.1)' if equity_df['cumulative_pnl'].iloc[-1] > 0 else 'rgba(255, 23, 68, 0.1)'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                xaxis_title="Trade Number",
                yaxis_title="Cumulative P&L ($)",
                height=450,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Additional Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L Distribution
                st.markdown("### 📊 P&L Distribution")
                
                fig = go.Figure()
                
                # Winning trades
                winning_trades = equity_df[equity_df['pnl'] > 0]
                if not winning_trades.empty:
                    fig.add_trace(go.Bar(
                        x=list(range(1, len(equity_df) + 1)),
                        y=[pnl if pnl > 0 else 0 for pnl in equity_df['pnl']],
                        name='Wins',
                        marker_color='#00c853'
                    ))
                
                # Losing trades
                losing_trades = equity_df[equity_df['pnl'] < 0]
                if not losing_trades.empty:
                    fig.add_trace(go.Bar(
                        x=list(range(1, len(equity_df) + 1)),
                        y=[pnl if pnl < 0 else 0 for pnl in equity_df['pnl']],
                        name='Losses',
                        marker_color='#ff1744'
                    ))
                
                fig.update_layout(
                    xaxis_title="Trade Number",
                    yaxis_title="P&L ($)",
                    height=350,
                    barmode='relative',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                
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
                        
                        fig.update_layout(
                            height=350,
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
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
                        
                        fig.update_layout(
                            height=350,
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 No closed trades with P&L data. Start trading and close positions to see your equity curve!")
        
        st.divider()
        
        # Recent Trades
        st.markdown("### 📋 Recent Trades")
        
        display_cols = []
        possible_cols = ['entry_date', 'symbol', 'side', 'outcome', 'quantity', 'entry_price', 'exit_price', 'pnl', 'status']
        for col in possible_cols:
            if col in df.columns:
                display_cols.append(col)
        
        if display_cols:
            recent_trades = df.sort_index(ascending=False).head(10)[display_cols].copy()
            
            # Format display
            if 'pnl' in recent_trades.columns:
                recent_trades['pnl'] = recent_trades['pnl'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "-"
                )
            
            st.dataframe(
                recent_trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "entry_date": st.column_config.DatetimeColumn("Entry Date", format="DD/MM/YYYY HH:mm"),
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "side": st.column_config.TextColumn("Side"),
                    "outcome": st.column_config.TextColumn("Outcome"),
                    "pnl": st.column_config.TextColumn("P&L"),
                    "status": st.column_config.TextColumn("Status"),
                }
            )
    else:
        st.info("📝 No trades recorded yet. Start by adding a new trade!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://via.placeholder.com/400x300?text=Start+Your+Trading+Journey", use_column_width=True)

# --- New Trade Page ---
elif page == "➕ New Trade":
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
            entry_time = st.time_input("Entry Time*", value=datetime.now().time())
        
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
            submitted = st.form_submit_button("💾 Save Trade", use_container_width=True, type="primary")
        
        if submitted:
            if symbol and quantity and entry_price:
                trade_data = {
                    "symbol": symbol.upper(),
                    "side": side,
                    "trade_type": trade_type,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": exit_price if exit_price > 0 else None,
                    "entry_date": datetime.combine(entry_date, entry_time).isoformat(),
                    "status": status,
                    "outcome": outcome,
                    "pnl": pnl if pnl != 0 else None,
                    "stop_loss": stop_loss if stop_loss > 0 else None,
                    "take_profit": take_profit if take_profit > 0 else None,
                    "risk_amount": risk_amount if risk_amount > 0 else None,
                    "risk_reward_ratio": risk_reward_ratio if risk_reward_ratio > 0 else None,
                    "strategy": strategy,
                    "timeframe": timeframe,
                    "entry_fee": entry_fee,
                    "exit_fee": exit_fee,
                    "total_fees": entry_fee + exit_fee,
                    "notes": notes,
                    "tags": tags,
                    "confidence_level": confidence_level,
                    "emotion": emotion,
                    "created_at": datetime.now().isoformat()
                }
                
                collection.insert_one(trade_data)
                st.success("✅ Trade saved successfully!")
                st.balloons()
            else:
                st.error("❌ Please fill in all required fields marked with *")

# --- Open Positions Page ---
elif page == "📈 Open Positions":
    st.title("📈 Open Positions")
    st.markdown("### Active trades currently in the market")
    
    docs = list(collection.find())
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
                        st.write(f"Entry Date: {trade.get('entry_date', 'N/A')}")
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
                                        "exit_price": exit_price,
                                        "exit_fee": exit_fee,
                                        "exit_date": datetime.now().isoformat(),
                                        "status": "CLOSED",
                                        "outcome": outcome,
                                        "pnl": pnl_input
                                    }
                                    
                                    collection.update_one(
                                        {"_id": trade['_id']},
                                        {"$set": update_data}
                                    )
                                    st.success("✅ Position closed!")
                                    st.rerun()
                            
                            with col_b:
                                if st.form_submit_button("🗑️ Delete", type="secondary", use_container_width=True):
                                    collection.delete_one({"_id": trade['_id']})
                                    st.success("🗑️ Trade deleted!")
                                    st.rerun()
                    
                    if trade.get('notes'):
                        st.divider()
                        st.markdown("**📝 Notes:**")
                        st.info(trade['notes'])
        else:
            st.info("📭 No open positions")
            st.image("https://via.placeholder.com/400x200?text=No+Open+Positions", use_column_width=True)
    else:
        st.info("📝 No trades recorded yet")

# --- Trade History Page ---
elif page == "📉 Trade History":
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
    
    # Load trades
    docs = list(collection.find(query))
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        
        # Display summary
        st.markdown(f"**Found {len(df)} trades**")
        
        st.divider()
        
        # Display columns selection
        available_cols = [col for col in df.columns if col != '_id']
        default_cols = ['entry_date', 'symbol', 'side', 'outcome', 'quantity', 'entry_price', 
                       'exit_price', 'pnl', 'status', 'strategy']
        display_cols = [col for col in default_cols if col in available_cols]
        
        selected_cols = st.multiselect(
            "📊 Select columns to display",
            available_cols,
            default=display_cols
        )
        
        if selected_cols:
            # Format the dataframe
            display_df = df[selected_cols].copy()
            
            # Add row numbers
            display_df.insert(0, '#', range(1, len(display_df) + 1))
            
            # Format numeric columns
            for col in ['entry_price', 'exit_price', 'stop_loss', 'take_profit']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) and x != 0 else "-"
                    )
            
            if 'pnl' in display_df.columns:
                display_df['pnl'] = display_df['pnl'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "-"
                )
            
            # Display the table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.divider()
            
            # Individual delete section - right below the table
            st.markdown("### 🗑️ Delete Individual Trade")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create readable options for trade selection
                trade_options = []
                for idx, row in df.iterrows():
                    entry_date = row.get('entry_date', 'N/A')
                    if isinstance(entry_date, str):
                        try:
                            entry_date = datetime.fromisoformat(entry_date).strftime('%Y-%m-%d')
                        except:
                            entry_date = 'N/A'
                    
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
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("🗑️ Delete Selected Trade", type="secondary", use_container_width=True):
                    if selected_trade_idx is not None:
                        trade_id = df.iloc[selected_trade_idx]['_id']
                        collection.delete_one({"_id": trade_id})
                        st.success("✅ Trade deleted successfully!")
                        st.rerun()
            
            st.divider()
            
            # Export and bulk delete buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                if st.button("🗑️ Delete All Filtered Trades", type="secondary", use_container_width=True):
                    st.warning("⚠️ Click again to confirm deletion of all filtered trades")
            
            with col3:
                # Confirmation for bulk delete
                if st.button("⚠️ Confirm Bulk Delete", type="secondary", use_container_width=True):
                    collection.delete_many(query)
                    st.success("✅ Filtered trades deleted!")
                    st.rerun()
                    
    else:
        st.info("📭 No trades found with the selected filters")
        st.image("https://via.placeholder.com/400x200?text=No+Trades+Found", use_column_width=True)

# --- Analytics Page ---
elif page == "📊 Analytics":
    st.title("📊 Trading Analytics")
    st.markdown("### Deep dive into your trading performance")
    
    docs = list(collection.find())
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
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(equity_df) + 1)),
                            y=equity_df['cumulative_pnl'],
                            mode='lines',
                            name='Cumulative P&L',
                            line=dict(
                                color='#00c853' if equity_df['cumulative_pnl'].iloc[-1] > 0 else '#ff1744',
                                width=3
                            ),
                            fill='tozeroy',
                            fillcolor='rgba(0, 200, 83, 0.1)' if equity_df['cumulative_pnl'].iloc[-1] > 0 else 'rgba(255, 23, 68, 0.1)'
                        ))
                        
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        fig.update_layout(
                            xaxis_title="Trade Number",
                            yaxis_title="Cumulative P&L ($)",
                            height=400,
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                        
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
                            marker=dict(color=colors)
                        ))
                        
                        fig.update_layout(
                            xaxis_title="P&L ($)",
                            yaxis_title="Symbol",
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                        
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
                                textposition='outside'
                            ))
                            
                            fig.update_layout(
                                xaxis_title="Win Rate (%)",
                                yaxis_title="Symbol",
                                height=400,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )
                            
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)', range=[0, 100])
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                            
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
                        fillcolor='rgba(255, 23, 68, 0.2)'
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Trade Number",
                        yaxis_title="Drawdown ($)",
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False
                    )
                    
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    
                    # Outcome distribution
                    if 'outcome' in trades_with_pnl.columns:
                        st.markdown("#### 🎯 Outcomes Distribution")
                        
                        outcome_data = trades_with_pnl.groupby('outcome').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).round(2)
                        
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
                            
                            fig.update_layout(
                                title="Trade Count by Outcome",
                                height=350,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            outcome_pnl = trades_with_pnl.groupby('outcome')['pnl'].sum()
                            
                            fig = go.Figure()
                            
                            colors = [colors_map.get(x, '#808080') for x in outcome_pnl.index]
                            
                            fig.add_trace(go.Bar(
                                x=outcome_pnl.index,
                                y=outcome_pnl.values,
                                marker=dict(color=colors)
                            ))
                            
                            fig.update_layout(
                                title="Total P&L by Outcome",
                                xaxis_title="Outcome",
                                yaxis_title="P&L ($)",
                                height=350,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False
                            )
                            
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if 'strategy' in trades_with_pnl.columns:
                        st.markdown("### Strategy Analysis")
                        
                        strategy_performance = trades_with_pnl.groupby('strategy').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).round(2)
                        
                        if not strategy_performance.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 📊 Trades by Strategy")
                                
                                strategy_counts = trades_with_pnl['strategy'].value_counts()
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=strategy_counts.index,
                                    values=strategy_counts.values,
                                    hole=.4
                                )])
                                
                                fig.update_layout(
                                    height=350,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                
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
                                    marker=dict(color=colors)
                                ))
                                
                                fig.update_layout(
                                    xaxis_title="Total P&L ($)",
                                    yaxis_title="Strategy",
                                    height=350,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    showlegend=False
                                )
                                
                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                
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
                    has_behavioral_data = False
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'emotion' in trades_with_pnl.columns:
                            st.markdown("#### 😊 P&L by Emotional State")
                            
                            emotion_performance = trades_with_pnl.groupby('emotion')['pnl'].mean().sort_values()
                            
                            if not emotion_performance.empty:
                                has_behavioral_data = True
                                
                                fig = go.Figure()
                                
                                colors = ['#00c853' if x > 0 else '#ff1744' for x in emotion_performance.values]
                                
                                fig.add_trace(go.Bar(
                                    y=emotion_performance.index,
                                    x=emotion_performance.values,
                                    orientation='h',
                                    marker=dict(color=colors)
                                ))
                                
                                fig.update_layout(
                                    xaxis_title="Average P&L ($)",
                                    yaxis_title="Emotional State",
                                    height=350,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    showlegend=False
                                )
                                
                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'confidence_level' in trades_with_pnl.columns:
                            st.markdown("#### 🎯 Confidence vs P&L")
                            
                            confidence_data = trades_with_pnl[['confidence_level', 'pnl']].dropna()
                            
                            if not confidence_data.empty:
                                has_behavioral_data = True
                                
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
                                
                                fig.update_layout(
                                    xaxis_title="Confidence Level",
                                    yaxis_title="P&L ($)",
                                    height=350,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    showlegend=False
                                )
                                
                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    if not has_behavioral_data:
                        st.info("📝 No behavioral data available. Add trades with emotion and confidence data to see analysis.")
                    
                    st.divider()
                    
                    # Time-based analysis
                    if 'entry_date' in trades_with_pnl.columns:
                        st.markdown("#### 📅 Performance Over Time")
                        
                        # Convert entry_date to datetime if it's not already
                        trades_with_pnl['entry_date'] = pd.to_datetime(trades_with_pnl['entry_date'])
                        
                        # Group by day
                        daily_pnl = trades_with_pnl.groupby(trades_with_pnl['entry_date'].dt.date)['pnl'].sum()
                        
                        fig = go.Figure()
                        
                        colors = ['#00c853' if x > 0 else '#ff1744' for x in daily_pnl.values]
                        
                        fig.add_trace(go.Bar(
                            x=daily_pnl.index,
                            y=daily_pnl.values,
                            marker=dict(color=colors)
                        ))
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Daily P&L ($)",
                            height=350,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        )
                        
                        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📝 No trades with P&L data. Add P&L to trades to see analytics.")
        else:
            st.info("📝 No closed trades available for analysis")
    else:
        st.info("📝 No data available for analytics")

# --- Settings Page ---
elif page == "⚙️ Settings":
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
                all_trades = list(collection.find())
                if all_trades:
                    df = pd.DataFrame(all_trades)
                    df['_id'] = df['_id'].astype(str)
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
                    st.rerun()
            
            st.divider()
            
            st.markdown("#### Delete All Open Trades")
            if st.checkbox("✅ I understand this will delete all OPEN trades"):
                if st.button("🗑️ Confirm Delete Open Trades", type="secondary"):
                    collection.delete_many({"status": "OPEN"})
                    st.success("✅ Open trades deleted")
                    st.rerun()
            
            st.divider()
            
            st.markdown("#### Delete All Closed Trades")
            if st.checkbox("✅ I understand this will delete all CLOSED trades"):
                if st.button("🗑️ Confirm Delete Closed Trades", type="secondary"):
                    collection.delete_many({"status": "CLOSED"})
                    st.success("✅ Closed trades deleted")
                    st.rerun()
    
    with tab2:
        st.markdown("### Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💱 Display Settings")
            currency = st.selectbox("Default Currency", ["USD", "EUR", "GBP", "JPY", "BTC", "ETH"])
            date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
            decimal_places = st.number_input("Decimal Places", min_value=0, max_value=8, value=2)
        
        with col2:
            st.markdown("#### 📊 Chart Settings")
            default_chart_height = st.number_input("Chart Height (px)", min_value=200, max_value=1000, value=400, step=50)
            show_grid = st.checkbox("Show Grid Lines", value=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("💾 Save Preferences", use_container_width=True, type="primary"):
                st.success("✅ Preferences saved!")
    
    with tab3:
        st.markdown("### About Trading Journal Pro")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://via.placeholder.com/200x200?text=TJP", use_column_width=True)
        
        with col2:
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
            - 📈 Equity curve visualization
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
            - 📱 Responsive design
            - 🔄 Backward compatibility
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
        
        st.markdown("### 📝 License")
        st.markdown("MIT License © 2024")
        
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

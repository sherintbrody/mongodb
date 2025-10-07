import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .success-metric {
        color: #00cc00;
    }
    .danger-metric {
        color: #ff0000;
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

# --- Helper Functions ---
def migrate_old_data(df):
    """Migrate old data format to new format"""
    if df.empty:
        return df
    
    # Add missing columns with default values
    if 'status' not in df.columns:
        df['status'] = 'CLOSED'  # Assume old trades are closed
    
    if 'entry_price' not in df.columns and 'price' in df.columns:
        df['entry_price'] = df['price']
    
    if 'quantity' not in df.columns and 'qty' in df.columns:
        df['quantity'] = df['qty']
    
    if 'side' in df.columns:
        # Convert old side format to new format
        df['side'] = df['side'].apply(lambda x: 'LONG' if x in ['BUY', 'LONG'] else 'SHORT')
    else:
        df['side'] = 'LONG'  # Default
    
    if 'entry_date' not in df.columns:
        df['entry_date'] = datetime.now()
    
    # Ensure numeric columns are numeric
    numeric_cols = ['quantity', 'entry_price', 'exit_price', 'stop_loss', 'take_profit', 'risk_amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_pnl(df):
    """Calculate P&L for closed trades"""
    if df.empty:
        return df
    
    df = migrate_old_data(df)
    
    # Only calculate P&L for closed trades with exit prices
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    
    if not closed_trades.empty and 'exit_price' in closed_trades.columns:
        # Calculate P&L based on side
        closed_trades['pnl'] = closed_trades.apply(
            lambda row: (row['exit_price'] - row['entry_price']) * row['quantity'] 
            if row['side'] == 'LONG' and pd.notna(row['exit_price'])
            else (row['entry_price'] - row['exit_price']) * row['quantity']
            if row['side'] == 'SHORT' and pd.notna(row['exit_price'])
            else 0,
            axis=1
        )
        
        closed_trades['pnl_percentage'] = closed_trades.apply(
            lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) 
            if row['side'] == 'LONG' and pd.notna(row['exit_price']) and row['entry_price'] != 0
            else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100)
            if row['side'] == 'SHORT' and pd.notna(row['exit_price']) and row['entry_price'] != 0
            else 0,
            axis=1
        )
        
        # Handle fees if they exist
        if 'total_fees' in closed_trades.columns:
            closed_trades['pnl_after_fees'] = closed_trades['pnl'] - closed_trades['total_fees'].fillna(0)
    
    return closed_trades

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
            'sharpe_ratio': 0
        }
    
    df = migrate_old_data(df)
    
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    
    if closed_trades.empty or 'exit_price' not in closed_trades.columns:
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
            'sharpe_ratio': 0
        }
    
    closed_trades = calculate_pnl(closed_trades)
    
    # Only calculate metrics if we have P&L data
    if 'pnl' not in closed_trades.columns:
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
            'sharpe_ratio': 0
        }
    
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] < 0]
    
    metrics = {
        'total_trades': len(closed_trades),
        'open_trades': len(df[df['status'] == 'OPEN']) if 'status' in df.columns else 0,
        'win_rate': len(winning_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0,
        'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
        'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
        'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else 0,
        'total_pnl': closed_trades['pnl'].sum() if not closed_trades.empty else 0,
        'best_trade': closed_trades['pnl'].max() if not closed_trades.empty else 0,
        'worst_trade': closed_trades['pnl'].min() if not closed_trades.empty else 0,
        'sharpe_ratio': closed_trades['pnl'].mean() / closed_trades['pnl'].std() if not closed_trades.empty and closed_trades['pnl'].std() != 0 else 0
    }
    
    return metrics

# --- Sidebar Navigation ---
st.sidebar.title("🚀 Trading Journal Pro")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "➕ New Trade", "📈 Open Positions", "📉 Trade History", 
     "📊 Analytics", "⚙️ Settings"]
)

# --- Dashboard Page ---
if page == "📊 Dashboard":
    st.title("📊 Trading Dashboard")
    
    # Load all trades
    docs = list(collection.find())
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        metrics = calculate_metrics(df)
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total P&L",
                f"${metrics['total_pnl']:.2f}",
                delta=f"{metrics['total_pnl']:.2f}" if metrics['total_pnl'] != 0 else None,
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1f}%",
                delta=f"{metrics['win_rate']-50:.1f}%" if metrics['win_rate'] != 0 else None
            )
        
        with col3:
            st.metric("Total Trades", metrics['total_trades'])
        
        with col4:
            st.metric("Open Positions", metrics['open_trades'])
        
        with col5:
            st.metric(
                "Profit Factor",
                f"{metrics['profit_factor']:.2f}",
                delta="Good" if metrics['profit_factor'] > 1.5 else "Poor"
            )
        
        st.divider()
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L Over Time
            closed_df = df[df['status'] == 'CLOSED'].copy()
            if not closed_df.empty and 'exit_price' in closed_df.columns:
                closed_df = calculate_pnl(closed_df)
                if 'pnl' in closed_df.columns and not closed_df['pnl'].isna().all():
                    closed_df['cumulative_pnl'] = closed_df['pnl'].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=closed_df.index,
                        y=closed_df['cumulative_pnl'],
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='green' if closed_df['cumulative_pnl'].iloc[-1] > 0 else 'red', width=2)
                    ))
                    fig.update_layout(
                        title="Cumulative P&L",
                        xaxis_title="Trade #",
                        yaxis_title="P&L ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No P&L data available yet. Close some trades to see P&L.")
            else:
                st.info("No closed trades with P&L data available.")
        
        with col2:
            # Trade Distribution by Symbol
            if 'symbol' in df.columns:
                symbol_counts = df['symbol'].value_counts()
                fig = px.pie(
                    values=symbol_counts.values,
                    names=symbol_counts.index,
                    title="Trade Distribution by Symbol"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent Trades
        st.subheader("📋 Recent Trades")
        # Display columns that exist
        display_cols = []
        possible_cols = ['entry_date', 'symbol', 'side', 'quantity', 'entry_price', 'status']
        for col in possible_cols:
            if col in df.columns:
                display_cols.append(col)
        
        if display_cols:
            recent_trades = df.sort_index(ascending=False).head(10)
            st.dataframe(
                recent_trades[display_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet. Start by adding a new trade!")

# --- New Trade Page ---
elif page == "➕ New Trade":
    st.title("➕ Add New Trade")
    
    with st.form("advanced_trade_form", clear_on_submit=True):
        st.subheader("Trade Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol*", placeholder="AAPL, BTC-USD, EUR/USD")
            side = st.selectbox("Side*", ["LONG", "SHORT"])
            trade_type = st.selectbox("Trade Type", ["STOCK", "FOREX", "CRYPTO", "OPTIONS", "FUTURES"])
        
        with col2:
            quantity = st.number_input("Quantity*", min_value=0.0001, step=0.01)
            entry_price = st.number_input("Entry Price*", min_value=0.0001, step=0.01)
            exit_price = st.number_input("Exit Price (if closed)", min_value=0.0, step=0.01)
        
        with col3:
            entry_date = st.date_input("Entry Date*", value=date.today())
            entry_time = st.time_input("Entry Time*", value=datetime.now().time())
            status = st.selectbox("Status*", ["OPEN", "CLOSED", "PENDING"])
        
        st.subheader("Risk Management")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)
        with col2:
            take_profit = st.number_input("Take Profit", min_value=0.0, step=0.01)
        with col3:
            risk_amount = st.number_input("Risk Amount ($)", min_value=0.0, step=1.0)
        with col4:
            risk_reward_ratio = st.number_input("R:R Ratio", min_value=0.0, step=0.1)
        
        st.subheader("Additional Information")
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
            entry_fee = st.number_input("Entry Fee ($)", min_value=0.0, step=0.01)
            exit_fee = st.number_input("Exit Fee ($)", min_value=0.0, step=0.01)
        
        notes = st.text_area("Trade Notes", placeholder="Entry reasons, market conditions, lessons learned...")
        
        tags = st.multiselect(
            "Tags",
            ["Earnings", "News", "Technical", "Fundamental", "FOMO", "Revenge Trade", "Plan Followed"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            confidence_level = st.slider("Confidence Level", 1, 10, 5)
        with col2:
            emotion = st.selectbox("Emotional State", ["Calm", "Excited", "Anxious", "Fearful", "Greedy"])
        
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
                st.error("Please fill in all required fields marked with *")

# --- Open Positions Page ---
elif page == "📈 Open Positions":
    st.title("📈 Open Positions")
    
    docs = list(collection.find())
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        open_df = df[df['status'] == 'OPEN']
        
        if not open_df.empty:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Positions", len(open_df))
            with col2:
                total_value = (open_df['quantity'] * open_df['entry_price']).sum()
                st.metric("Total Value", f"${total_value:.2f}")
            with col3:
                unique_symbols = open_df['symbol'].nunique()
                st.metric("Unique Symbols", unique_symbols)
            
            st.divider()
            
            # Display open positions with actions
            for idx, trade in open_df.iterrows():
                with st.expander(f"{trade['symbol']} - {trade['side']} - {trade['quantity']} @ ${trade['entry_price']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Entry Date:** {trade.get('entry_date', 'N/A')}")
                        st.write(f"**Strategy:** {trade.get('strategy', 'N/A')}")
                        st.write(f"**Stop Loss:** ${trade.get('stop_loss', 'Not set')}")
                    
                    with col2:
                        st.write(f"**Take Profit:** ${trade.get('take_profit', 'Not set')}")
                        st.write(f"**Risk Amount:** ${trade.get('risk_amount', 'N/A')}")
                        st.write(f"**Timeframe:** {trade.get('timeframe', 'N/A')}")
                    
                    with col3:
                        # Close position form
                        with st.form(f"close_{trade['_id']}"):
                            exit_price = st.number_input("Exit Price", min_value=0.01, key=f"exit_{trade['_id']}")
                            exit_fee = st.number_input("Exit Fee", min_value=0.0, key=f"fee_{trade['_id']}")
                            if st.form_submit_button("Close Position"):
                                collection.update_one(
                                    {"_id": trade['_id']},
                                    {"$set": {
                                        "exit_price": exit_price,
                                        "exit_fee": exit_fee,
                                        "exit_date": datetime.now().isoformat(),
                                        "status": "CLOSED"
                                    }}
                                )
                                st.success("Position closed!")
                                st.rerun()
        else:
            st.info("No open positions")
    else:
        st.info("No trades recorded yet")

# --- Trade History Page ---
elif page == "📉 Trade History":
    st.title("📉 Trade History")
    
    # Filters
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_symbol = st.text_input("Symbol", placeholder="All")
        with col2:
            filter_status = st.selectbox("Status", ["All", "OPEN", "CLOSED", "PENDING"])
        with col3:
            filter_side = st.selectbox("Side", ["All", "LONG", "SHORT", "BUY", "SELL"])
        with col4:
            filter_strategy = st.selectbox(
                "Strategy",
                ["All", "Scalping", "Day Trading", "Swing Trading", "Position Trading", 
                 "Momentum", "Mean Reversion", "Breakout", "Other"]
            )
    
    # Build query
    query = {}
    if filter_symbol:
        query["symbol"] = {"$regex": filter_symbol.upper()}
    if filter_status != "All":
        query["status"] = filter_status
    if filter_side != "All":
        if filter_side in ["BUY", "SELL"]:
            query["side"] = filter_side
        else:
            query["side"] = filter_side
    if filter_strategy != "All":
        query["strategy"] = filter_strategy
    
    # Load trades
    docs = list(collection.find(query))
    
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        
        # Calculate P&L for closed trades
        if 'status' in df.columns:
            closed_df = df[df['status'] == 'CLOSED'].copy()
            if not closed_df.empty and 'exit_price' in closed_df.columns:
                closed_df = calculate_pnl(closed_df)
                if 'pnl' in closed_df.columns:
                    df = df.merge(
                        closed_df[['_id', 'pnl', 'pnl_percentage']],
                        on='_id',
                        how='left'
                    )
        
        # Display columns selection
        available_cols = [col for col in df.columns if col != '_id']
        default_cols = ['entry_date', 'symbol', 'side', 'quantity', 'entry_price', 
                       'exit_price', 'status', 'pnl', 'strategy']
        display_cols = [col for col in default_cols if col in available_cols]
        
        selected_cols = st.multiselect(
            "Select columns to display",
            available_cols,
            default=display_cols
        )
        
        if selected_cols:
            # Format the dataframe
            display_df = df[selected_cols].copy()
            
            # Format numeric columns
            for col in ['entry_price', 'exit_price', 'pnl', 'stop_loss', 'take_profit']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) and x != 0 else "-"
                    )
            
            if 'pnl_percentage' in display_df.columns:
                display_df['pnl_percentage'] = display_df['pnl_percentage'].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
                )
            
            # Display the table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No trades found with the selected filters")

# --- Analytics Page ---
elif page == "📊 Analytics":
    st.title("📊 Trading Analytics")
    
    docs = list(collection.find())
    if docs:
        df = pd.DataFrame(docs)
        df = migrate_old_data(df)
        
        # Only show analytics for closed trades with P&L data
        closed_df = df[df['status'] == 'CLOSED'].copy()
        
        if not closed_df.empty and 'exit_price' in closed_df.columns:
            closed_df = calculate_pnl(closed_df)
            
            if 'pnl' in closed_df.columns and not closed_df['pnl'].isna().all():
                # Tabs for different analytics
                tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Strategy Analysis", "Behavioral Analysis"])
                
                with tab1:
                    st.subheader("Performance Metrics")
                    
                    # Performance by Symbol
                    symbol_performance = closed_df.groupby('symbol').agg({
                        'pnl': ['sum', 'mean', 'count']
                    }).round(2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            x=symbol_performance.index,
                            y=symbol_performance[('pnl', 'sum')],
                            title="Total P&L by Symbol",
                            labels={'y': 'P&L ($)', 'x': 'Symbol'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Win rate by symbol
                        win_rates = []
                        for symbol in closed_df['symbol'].unique():
                            symbol_trades = closed_df[closed_df['symbol'] == symbol]
                            wins = len(symbol_trades[symbol_trades['pnl'] > 0])
                            total = len(symbol_trades)
                            win_rates.append({
                                'symbol': symbol,
                                'win_rate': (wins/total * 100) if total > 0 else 0
                            })
                        
                        if win_rates:
                            win_rate_df = pd.DataFrame(win_rates)
                            fig = px.bar(
                                win_rate_df,
                                x='symbol',
                                y='win_rate',
                                title="Win Rate by Symbol",
                                labels={'win_rate': 'Win Rate (%)', 'symbol': 'Symbol'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Risk Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        max_drawdown = closed_df['pnl'].cumsum().min()
                        st.metric("Max Drawdown", f"${max_drawdown:.2f}")
                    
                    with col2:
                        if 'risk_amount' in closed_df.columns:
                            avg_risk = closed_df['risk_amount'].mean()
                            st.metric("Average Risk", f"${avg_risk:.2f}")
                    
                    with col3:
                        if 'risk_reward_ratio' in closed_df.columns:
                            avg_rr = closed_df['risk_reward_ratio'].mean()
                            st.metric("Avg R:R Ratio", f"{avg_rr:.2f}")
                
                with tab3:
                    if 'strategy' in closed_df.columns:
                        st.subheader("Strategy Analysis")
                        strategy_performance = closed_df.groupby('strategy').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).round(2)
                        
                        if not strategy_performance.empty:
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.pie(
                                    values=strategy_performance[('pnl', 'count')],
                                    names=strategy_performance.index,
                                    title="Trades by Strategy"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig = px.bar(
                                    x=strategy_performance.index,
                                    y=strategy_performance[('pnl', 'sum')],
                                    title="P&L by Strategy",
                                    labels={'y': 'Total P&L ($)', 'x': 'Strategy'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No strategy data available")
                
                with tab4:
                    st.subheader("Behavioral Analysis")
                    has_behavioral_data = False
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'emotion' in closed_df.columns:
                            emotion_performance = closed_df.groupby('emotion')['pnl'].mean()
                            if not emotion_performance.empty:
                                has_behavioral_data = True
                                fig = px.bar(
                                    x=emotion_performance.index,
                                    y=emotion_performance.values,
                                    title="Average P&L by Emotional State",
                                    labels={'y': 'Avg P&L ($)', 'x': 'Emotion'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'confidence_level' in closed_df.columns:
                            confidence_data = closed_df[['confidence_level', 'pnl']].dropna()
                            if not confidence_data.empty:
                                has_behavioral_data = True
                                fig = px.scatter(
                                    confidence_data,
                                    x='confidence_level',
                                    y='pnl',
                                    title="P&L vs Confidence Level",
                                    labels={'confidence_level': 'Confidence Level', 'pnl': 'P&L ($)'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    if not has_behavioral_data:
                        st.info("No behavioral data available. Add trades with emotion and confidence data to see analysis.")
            else:
                st.info("No trades with complete P&L data. Close some trades with exit prices to see analytics.")
        else:
            st.info("No closed trades available for analysis")
    else:
        st.info("No data available for analytics")

# --- Settings Page ---
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Database", "Preferences", "About"])
    
    with tab1:
        st.subheader("Database Management")
        
        # Database stats
        total_trades = collection.count_documents({})
        st.metric("Total Records", total_trades)
        
        # Backup
        if st.button("📥 Backup Database"):
            all_trades = list(collection.find())
            if all_trades:
                df = pd.DataFrame(all_trades)
                # Convert ObjectId to string for CSV export
                df['_id'] = df['_id'].astype(str)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Backup",
                    data=csv,
                    file_name=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Danger Zone
        st.divider()
        st.subheader("⚠️ Danger Zone")
        with st.expander("Delete Operations", expanded=False):
            if st.button("Delete All Trades", type="secondary"):
                if st.checkbox("I understand this will delete all trades"):
                    if st.button("Confirm Delete All"):
                        collection.delete_many({})
                        st.success("All trades deleted")
                        st.rerun()
    
    with tab2:
        st.subheader("Preferences")
        
        # Theme preference (for demonstration)
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        currency = st.selectbox("Default Currency", ["USD", "EUR", "GBP", "JPY", "BTC"])
        date_format = st.selectbox("Date Format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"])
        
        if st.button("Save Preferences"):
            st.success("Preferences saved!")
    
    with tab3:
        st.subheader("About")
        st.info("""
        **Trading Journal Pro v2.0**
        
        A comprehensive trading journal application built with Streamlit and MongoDB.
        
        **Features:**
        - Track trades across multiple asset classes
        - Advanced analytics and performance metrics
        - Risk management tools
        - Strategy analysis
        - Behavioral tracking
        - Backward compatible with old data format
        
        **Created with:** Streamlit, MongoDB, Plotly, Pandas
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #888'>Trading Journal Pro © 2024 | Trade Responsibly</p>
    </div>
    """,
    unsafe_allow_html=True
)

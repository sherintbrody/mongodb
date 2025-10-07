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
def calculate_pnl(df):
    """Calculate P&L for closed trades"""
    if df.empty:
        return df
    
    # Group by symbol and calculate P&L
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    if not closed_trades.empty:
        closed_trades['pnl'] = closed_trades.apply(
            lambda row: (row['exit_price'] - row['entry_price']) * row['quantity'] 
            if row['side'] == 'LONG' 
            else (row['entry_price'] - row['exit_price']) * row['quantity'],
            axis=1
        )
        closed_trades['pnl_percentage'] = closed_trades.apply(
            lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) 
            if row['side'] == 'LONG' 
            else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100),
            axis=1
        )
        closed_trades['pnl_after_fees'] = closed_trades['pnl'] - closed_trades.get('total_fees', 0)
    return closed_trades

def calculate_metrics(df):
    """Calculate trading metrics"""
    if df.empty:
        return {}
    
    closed_trades = df[df['status'] == 'CLOSED'].copy()
    if closed_trades.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0
        }
    
    closed_trades = calculate_pnl(closed_trades)
    
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] < 0]
    
    metrics = {
        'total_trades': len(closed_trades),
        'open_trades': len(df[df['status'] == 'OPEN']),
        'win_rate': len(winning_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0,
        'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
        'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
        'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else 0,
        'total_pnl': closed_trades['pnl'].sum() if not closed_trades.empty else 0,
        'best_trade': closed_trades['pnl'].max() if not closed_trades.empty else 0,
        'worst_trade': closed_trades['pnl'].min() if not closed_trades.empty else 0,
        'avg_holding_time': 0,  # Would need exit_date to calculate
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
            if not closed_df.empty:
                closed_df = calculate_pnl(closed_df)
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
        
        with col2:
            # Win/Loss Distribution
            if not closed_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=closed_df['pnl'],
                    name='P&L Distribution',
                    marker_color=['green' if x > 0 else 'red' for x in closed_df['pnl']],
                    nbinsx=20
                ))
                fig.update_layout(
                    title="P&L Distribution",
                    xaxis_title="P&L ($)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent Trades
        st.subheader("📋 Recent Trades")
        recent_trades = df.sort_values('entry_date', ascending=False).head(10)
        display_cols = ['entry_date', 'symbol', 'side', 'quantity', 'entry_price', 'status']
        if all(col in recent_trades.columns for col in display_cols):
            st.dataframe(
                recent_trades[display_cols],
                use_container_width=True,
                hide_index=True
            )
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
    
    docs = list(collection.find({"status": "OPEN"}))
    if docs:
        df = pd.DataFrame(docs)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Open Positions", len(df))
        with col2:
            total_value = (df['quantity'] * df['entry_price']).sum()
            st.metric("Total Value", f"${total_value:.2f}")
        with col3:
            unique_symbols = df['symbol'].nunique()
            st.metric("Unique Symbols", unique_symbols)
        
        st.divider()
        
        # Display open positions with actions
        for idx, trade in df.iterrows():
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
            filter_side = st.selectbox("Side", ["All", "LONG", "SHORT"])
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
        query["side"] = filter_side
    if filter_strategy != "All":
        query["strategy"] = filter_strategy
    
    # Load trades
    docs = list(collection.find(query).sort("entry_date", -1))
    
    if docs:
        df = pd.DataFrame(docs)
        
        # Calculate P&L for closed trades
        if 'status' in df.columns:
            closed_df = calculate_pnl(df)
            if not closed_df.empty and 'pnl' in closed_df.columns:
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
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
            
            if 'pnl_percentage' in display_df.columns:
                display_df['pnl_percentage'] = display_df['pnl_percentage'].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
                )
            
            # Display the table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pnl": st.column_config.NumberColumn(
                        "P&L",
                        format="$%.2f"
                    ),
                }
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
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=date.today())
        
        # Filter data by date if entry_date exists
        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            mask = (df['entry_date'].dt.date >= start_date) & (df['entry_date'].dt.date <= end_date)
            df = df.loc[mask]
        
        if not df.empty:
            # Tabs for different analytics
            tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Analysis", "Strategy Analysis", "Behavioral Analysis"])
            
            with tab1:
                st.subheader("Performance Metrics")
                
                closed_df = calculate_pnl(df)
                if not closed_df.empty:
                    # Performance by Symbol
                    symbol_performance = closed_df.groupby('symbol').agg({
                        'pnl': ['sum', 'mean', 'count'],
                        'pnl_percentage': 'mean'
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
                        
                        win_rate_df = pd.DataFrame(win_rates)
                        fig = px.bar(
                            win_rate_df,
                            x='symbol',
                            y='win_rate',
                            title="Win Rate by Symbol",
                            labels={'win_rate': 'Win Rate (%)', 'symbol': 'Symbol'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly Performance
                    if 'entry_date' in closed_df.columns:
                        closed_df['month'] = closed_df['entry_date'].dt.to_period('M')
                        monthly_pnl = closed_df.groupby('month')['pnl'].sum()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=monthly_pnl.index.astype(str),
                            y=monthly_pnl.values,
                            marker_color=['green' if x > 0 else 'red' for x in monthly_pnl.values],
                            text=monthly_pnl.values.round(2),
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title="Monthly P&L",
                            xaxis_title="Month",
                            yaxis_title="P&L ($)",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Risk Analysis")
                
                if not closed_df.empty:
                    # Risk metrics
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
                    
                    # Risk distribution
                    if 'risk_amount' in closed_df.columns:
                        fig = px.histogram(
                            closed_df,
                            x='risk_amount',
                            title="Risk Amount Distribution",
                            labels={'risk_amount': 'Risk Amount ($)'},
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Strategy Analysis")
                
                if 'strategy' in df.columns:
                    strategy_df = calculate_pnl(df[df['status'] == 'CLOSED'])
                    if not strategy_df.empty:
                        strategy_performance = strategy_df.groupby('strategy').agg({
                            'pnl': ['sum', 'mean', 'count']
                        }).round(2)
                        
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
            
            with tab4:
                st.subheader("Behavioral Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'emotion' in df.columns:
                        emotion_df = calculate_pnl(df[df['status'] == 'CLOSED'])
                        if not emotion_df.empty and 'emotion' in emotion_df.columns:
                            emotion_performance = emotion_df.groupby('emotion')['pnl'].mean()
                            fig = px.bar(
                                x=emotion_performance.index,
                                y=emotion_performance.values,
                                title="Average P&L by Emotional State",
                                labels={'y': 'Avg P&L ($)', 'x': 'Emotion'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'confidence_level' in df.columns:
                        confidence_df = calculate_pnl(df[df['status'] == 'CLOSED'])
                        if not confidence_df.empty and 'confidence_level' in confidence_df.columns:
                            fig = px.scatter(
                                confidence_df,
                                x='confidence_level',
                                y='pnl',
                                title="P&L vs Confidence Level",
                                labels={'confidence_level': 'Confidence Level', 'pnl': 'P&L ($)'},
                                trendline="ols"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Trading frequency analysis
                if 'entry_date' in df.columns:
                    df['weekday'] = df['entry_date'].dt.day_name()
                    df['hour'] = df['entry_date'].dt.hour
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        weekday_counts = df['weekday'].value_counts()
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        weekday_counts = weekday_counts.reindex(day_order, fill_value=0)
                        
                        fig = px.bar(
                            x=weekday_counts.index,
                            y=weekday_counts.values,
                            title="Trading Activity by Day of Week",
                            labels={'x': 'Day', 'y': 'Number of Trades'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        hour_counts = df['hour'].value_counts().sort_index()
                        fig = px.line(
                            x=hour_counts.index,
                            y=hour_counts.values,
                            title="Trading Activity by Hour",
                            labels={'x': 'Hour', 'y': 'Number of Trades'},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
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

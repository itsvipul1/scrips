import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 Positional Portfolio Dashboard")

# Replace this string with your published Google Sheet CSV link
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD/pub?output=csv"

@st.cache_data(ttl=3600)
def load_portfolio():
    return pd.read_csv(SHEET_CSV_URL)

@st.cache_data(ttl=3600)
def fetch_all_stock_data(symbols):
    # Fetch 5 years of history to support long-term weekly charts and moving averages
    data = yf.download(symbols, period="5y", threads=True, progress=False)
    return data

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- LOAD DATA ---
try:
    portfolio = load_portfolio()
    symbols_list = portfolio['Symbol'].dropna().unique().tolist()
except Exception as e:
    st.error("Error loading Google Sheet. Please check the link.")
    st.stop()

if not symbols_list:
    st.warning("No symbols found in the Google Sheet.")
    st.stop()

# --- DASHBOARD UI ---
st.write("### Market Overview")

# Top Control Panel
col_t1, col_t2 = st.columns(2)
with col_t1:
    # Increased slider range up to 5 years (1825 days), default set to 3 years (1095 days)
    days_to_plot = st.slider("Select chart visual timeframe (Days)", min_value=30, max_value=1825, value=1095)
with col_t2:
    chart_type = st.radio("Chart Timeframe", ["Daily", "Weekly"], horizontal=True)

with st.spinner('Fetching bulk market data from Yahoo Finance...'):
    market_data = fetch_all_stock_data(symbols_list)

# --- SUMMARY TABLE ---
summary_data = []
for index, row in portfolio.iterrows():
    symbol = row['Symbol']
    
    try:
        purchased_at = float(row.get('PurchasedAt', 0))
        if purchased_at <= 0:
            continue
    except (ValueError, TypeError):
        continue
        
    try:
        if isinstance(market_data.columns, pd.MultiIndex):
            if symbol in market_data.columns.levels[1]:
                cp_raw = market_data['Close'][symbol].dropna().iloc[-1]
            else:
                cp_raw = market_data.xs(symbol, level=0, axis=1)['Close'].dropna().iloc[-1]
        else:
            cp_raw = market_data['Close'].dropna().iloc[-1]
            
        if isinstance(cp_raw, (pd.Series, pd.DataFrame)):
            cp_raw = cp_raw.iloc[-1]
        current_price = float(cp_raw)
        
        pct_change = ((current_price - purchased_at) / purchased_at) * 100
        
        summary_data.append({
            "Symbol": symbol.replace('.NS', ''),
            "Purchased At": round(purchased_at, 2),
            "Current Price": round(current_price, 2),
            "% Return": round(pct_change, 2)
        })
    except Exception:
        pass

# Render the Table
if summary_data:
    st.write("#### 🎯 Portfolio Performance")
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="% Return", ascending=False)
    
    st.dataframe(
        summary_df,
        column_config={
            "% Return": st.column_config.NumberColumn("% Return", format="%.2f %%"),
            "Purchased At": st.column_config.NumberColumn("Purchased At", format="₹%.2f"),
            "Current Price": st.column_config.NumberColumn("Current Price", format="₹%.2f")
        },
        use_container_width=True,
        hide_index=True
    )

# --- INDIVIDUAL CHARTS ---
st.write("#### 📊 Technical Charts")

start_plot_date = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=days_to_plot))

for index, row in portfolio.iterrows():
    symbol = row['Symbol']
    
    try:
        target = float(row.get('Target', 0))
        stop_loss = float(row.get('StopLoss', 0))
        purchased_at = float(row.get('PurchasedAt', 0))
    except ValueError:
        st.warning(f"Check your Google Sheet: Target, StopLoss, or PurchasedAt for {symbol} is not a valid number.")
        continue
    
    st.markdown("---")
    
    try:
        if isinstance(market_data.columns, pd.MultiIndex):
            if symbol in market_data.columns.levels[1]:
                df = market_data.xs(symbol, level=1, axis=1).copy()
            elif symbol in market_data.columns.levels[0]:
                df = market_data.xs(symbol, level=0, axis=1).copy()
            else:
                continue
        else:
            df = market_data.copy()
    except Exception:
        continue

    df = df.dropna(how='all')
    
    if df.empty or ('Close' not in df.columns) or df['Close'].isna().all():
        continue

    for col in ['Open', 'High', 'Low', 'Close']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    if chart_type == "Weekly":
        df = df.resample('W-FRI').agg({
            'Open': 'first', 
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last'
        }).dropna()

    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    plot_df = df[df.index >= start_plot_date]
    
    if plot_df.empty:
        st.warning(f"No data for {symbol} in the selected timeframe.")
        continue
        
    current_price_raw = plot_df['Close'].iloc[-1]
    if isinstance(current_price_raw, (pd.Series, pd.DataFrame)):
        current_price_raw = current_price_raw.iloc[-1]
    current_price = float(current_price_raw)
    
    pct_to_target = ((target - current_price) / current_price) * 100 if target > 0 else 0
    pct_to_stop = ((current_price - stop_loss) / current_price) * 100 if stop_loss > 0 else 0
    pct_return = ((current_price - purchased_at) / purchased_at) * 100 if purchased_at > 0 else 0

    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader(symbol.replace('.NS', ''))
        st.metric("Current Price", f"₹{current_price:.2f}")
        if purchased_at > 0:
            st.metric("Purchased At", f"₹{purchased_at:.2f}", f"{pct_return:+.1f}% Return")
        if target > 0:
            st.metric("Target", f"₹{target:.2f}", f"{pct_to_target:.1f}% away")
        if stop_loss > 0:
            st.metric("Stop Loss", f"₹{stop_loss:.2f}", f"-{pct_to_stop:.1f}% risk", delta_color="inverse")

    with col2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # 1. Candlestick
        fig.add_trace(go.Candlestick(x=plot_df.index,
                                     open=plot_df['Open'], high=plot_df['High'], 
                                     low=plot_df['Low'], close=plot_df['Close'],
                                     name='Price'), row=1, col=1)
        
        # 2. Moving Average
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['50_MA'], 
                                 line=dict(color='blue', width=1.5), name='50 MA'), row=1, col=1)
        
        # 3. Horizontal Lines & Legend Entries
        if target > 0:
            fig.add_trace(go.Scatter(x=plot_df.index, y=[target]*len(plot_df), 
                                     line=dict(color='green', width=2, dash='dash'), name='Target'), row=1, col=1)
        if stop_loss > 0:
            fig.add_trace(go.Scatter(x=plot_df.index, y=[stop_loss]*len(plot_df), 
                                     line=dict(color='red', width=2, dash='dash'), name='Stop Loss'), row=1, col=1)
        if purchased_at > 0:
            # Invisible ghost trace just to insert the Purchase Price into the legend
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(color='rgba(0,0,0,0)'), 
                                     name=f'Purchased @ ₹{purchased_at:.2f}'), row=1, col=1)

        # 4. RSI Chart
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], 
                                 line=dict(color='orange', width=1.5), name='RSI'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=plot_df.index, y=[70]*len(plot_df), 
                                 line=dict(color='gray', width=1, dash='dash'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=[30]*len(plot_df), 
                                 line=dict(color='gray', width=1, dash='dash'), showlegend=False), row=2, col=1)
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), 
                          xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

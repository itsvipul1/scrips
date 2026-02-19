import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 Positional Portfolio Dashboard")

# Replace this string with your published Google Sheet CSV link
SHEET_CSV_URL = "YOUR_GOOGLE_SHEET_CSV_LINK_HERE"

@st.cache_data(ttl=3600)
def load_portfolio():
    return pd.read_csv(SHEET_CSV_URL)

# NEW BATCH FETCH FUNCTION
@st.cache_data(ttl=3600)
def fetch_all_stock_data(symbols, days=100):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days + 50) # Extra days for 50-DMA
    
    # Download all symbols at once. 'threads=True' makes it lightning fast.
    data = yf.download(symbols, start=start_date, end=end_date, threads=True, progress=False)
    return data

# --- LOAD DATA ---
try:
    portfolio = load_portfolio()
    # Create a clean list of unique symbols from your sheet
    symbols_list = portfolio['Symbol'].dropna().unique().tolist()
except Exception as e:
    st.error("Error loading Google Sheet. Please check the link.")
    st.stop()

if not symbols_list:
    st.warning("No symbols found in the Google Sheet.")
    st.stop()

# --- DASHBOARD UI ---
st.write("### Market Overview")
days_to_plot = st.slider("Select chart timeframe (Days)", min_value=30, max_value=200, value=100)

# Fetch ALL data in one single API call
with st.spinner('Fetching bulk market data from Yahoo Finance...'):
    market_data = fetch_all_stock_data(symbols_list, days_to_plot)

# Loop through each stock in your Google Sheet to render the charts
for index, row in portfolio.iterrows():
    symbol = row['Symbol']
    target = row['Target']
    stop_loss = row['StopLoss']
    
    st.markdown(f"---")
    
    try:
        # Slice the large batch dataset for this specific symbol
        if isinstance(market_data.columns, pd.MultiIndex):
            # yfinance returns a MultiIndex when downloading multiple tickers
            df = market_data.xs(symbol, level=1, axis=1).copy()
        else:
            # Fallback if there is only 1 ticker in the sheet
            df = market_data.copy()
    except KeyError:
        st.warning(f"Could not find data for {symbol} in the downloaded batch. Check the symbol.")
        continue

    # Clean the data (drop weekend/holiday empty rows)
    df = df.dropna(how='all')
    
    if df.empty or ('Close' in df.columns and df['Close'].isna().all()):
        st.warning(f"No valid price data available for {symbol}.")
        continue

    # Calculate 50-Day Moving Average
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Slice dataframe to the user's selected timeframe
    plot_df = df.tail(days_to_plot)
    
    if plot_df.empty:
        continue
        
    # Force current_price to be a standard float to prevent formatting errors
    current_price = float(plot_df['Close'].iloc[-1])
    
    # Calculate percentages
    pct_to_target = ((target - current_price) / current_price) * 100
    pct_to_stop = ((current_price - stop_loss) / current_price) * 100

    # Layout: Metrics on the left, Chart on the right
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader(symbol.replace('.NS', ''))
        st.metric("Current Price", f"₹{current_price:.2f}")
        st.metric("Target", f"₹{target:.2f}", f"{pct_to_target:.1f}% away")
        st.metric("Stop Loss", f"₹{stop_loss:.2f}", f"-{pct_to_stop:.1f}% risk", delta_color="inverse")

    with col2:
        # Create interactive Plotly Candlestick Chart
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(x=plot_df.index,
                                     open=plot_df['Open'],
                                     high=plot_df['High'],
                                     low=plot_df['Low'],
                                     close=plot_df['Close'],
                                     name='Price'))
        
        # 50 DMA
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['50_MA'], 
                                 line=dict(color='blue', width=1.5), name='50 DMA'))
        
        # Target Line
        fig.add_trace(go.Scatter(x=plot_df.index, y=[target]*len(plot_df), 
                                 line=dict(color='green', width=2, dash='dash'), name='Target'))
        
        # Stop Loss Line
        fig.add_trace(go.Scatter(x=plot_df.index, y=[stop_loss]*len(plot_df), 
                                 line=dict(color='red', width=2, dash='dash'), name='Stop Loss'))

        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), 
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 Positional Portfolio Dashboard")

# Replace this string with your published Google Sheet CSV link
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD/pubhtml"

@st.cache_data(ttl=3600)
def load_portfolio():
    return pd.read_csv(SHEET_CSV_URL)

# NEW BATCH FETCH FUNCTION
@st.cache_data(ttl=3600)
def fetch_all_stock_data(symbols, days=100):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days + 50) 
    data = yf.download(symbols, start=start_date, end=end_date, threads=True, progress=False)
    return data

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
days_to_plot = st.slider("Select chart timeframe (Days)", min_value=30, max_value=200, value=100)

with st.spinner('Fetching bulk market data from Yahoo Finance...'):
    market_data = fetch_all_stock_data(symbols_list, days_to_plot)

# Loop through each stock in your Google Sheet
for index, row in portfolio.iterrows():
    symbol = row['Symbol']
    
    # 1. Force target and stop loss to be pure numbers (in case the sheet formats them as text)
    try:
        target = float(row['Target'])
        stop_loss = float(row['StopLoss'])
    except ValueError:
        st.warning(f"Check your Google Sheet: Target or StopLoss for {symbol} is not a valid number.")
        continue
    
    st.markdown("---")
    
    try:
        # 2. Slice the large batch dataset safely
        if isinstance(market_data.columns, pd.MultiIndex):
            # Check which level the ticker is on (yfinance changes this depending on the version)
            if symbol in market_data.columns.levels[1]:
                df = market_data.xs(symbol, level=1, axis=1).copy()
            elif symbol in market_data.columns.levels[0]:
                df = market_data.xs(symbol, level=0, axis=1).copy()
            else:
                st.warning(f"Could not find data for {symbol}.")
                continue
        else:
            df = market_data.copy()
    except Exception:
        st.warning(f"Error processing data for {symbol}.")
        continue

    # Clean the data
    df = df.dropna(how='all')
    
    if df.empty or ('Close' not in df.columns) or df['Close'].isna().all():
        st.warning(f"No valid price data available for {symbol}.")
        continue

    # 3. Force Close to be a 1-Dimensional Series (fixes overlapping columns)
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]

    # Calculate 50-Day Moving Average
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Slice dataframe to the user's selected timeframe
    plot_df = df.tail(days_to_plot)
    
    if plot_df.empty:
        continue
        
    # 4. ABSOLUTE BULLETPROOF PRICE EXTRACTION
    current_price_raw = plot_df['Close'].iloc[-1]
    # Keep unwrapping the data until it's a raw number
    if isinstance(current_price_raw, (pd.Series, pd.DataFrame)):
        current_price_raw = current_price_raw.iloc[-1]
    
    current_price = float(current_price_raw)
    
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
        fig = go.Figure()
        
        # Safely extract open, high, low for candlestick chart
        open_p = plot_df['Open'].iloc[:, 0] if isinstance(plot_df['Open'], pd.DataFrame) else plot_df['Open']
        high_p = plot_df['High'].iloc[:, 0] if isinstance(plot_df['High'], pd.DataFrame) else plot_df['High']
        low_p = plot_df['Low'].iloc[:, 0] if isinstance(plot_df['Low'], pd.DataFrame) else plot_df['Low']
        close_p = plot_df['Close']
        
        fig.add_trace(go.Candlestick(x=plot_df.index,
                                     open=open_p, high=high_p, low=low_p, close=close_p,
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


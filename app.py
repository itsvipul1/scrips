import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 Positional Trading Dashboard")

# ⚠️ REPLACE THESE TWO STRINGS WITH YOUR GOOGLE SHEET CSV LINKS
PORTFOLIO_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD/pub?gid=0&single=true&output=csv"
WATCHLIST_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD/pub?gid=186620296&single=true&output=csv"

@st.cache_data(ttl=3600)
def load_csv(url):
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_all_stock_data(symbols):
    data = yf.download(symbols, period="5y", threads=True, progress=False)
    return data

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- LOAD DATA ---
portfolio = load_csv(PORTFOLIO_CSV_URL)
watchlist = load_csv(WATCHLIST_CSV_URL)

port_symbols = portfolio['Symbol'].dropna().unique().tolist() if not portfolio.empty else []
watch_symbols = watchlist['Symbol'].dropna().unique().tolist() if not watchlist.empty else []
all_symbols = list(set(port_symbols + watch_symbols))

if not all_symbols:
    st.error("No symbols found. Please check your Google Sheet CSV links.")
    st.stop()

# --- TOP CONTROL PANEL ---
col_t1, col_t2 = st.columns(2)
with col_t1:
    days_to_plot = st.slider("Select chart visual timeframe (Days)", min_value=30, max_value=1825, value=1095)
with col_t2:
    chart_type = st.radio("Chart Timeframe", ["Daily", "Weekly"], horizontal=True)

with st.spinner('Fetching bulk market data from Yahoo Finance...'):
    market_data = fetch_all_stock_data(all_symbols)

start_plot_date = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=days_to_plot))

# --- HELPER FUNCTION: RENDER CHARTS ---
def render_stock_row(row, df, mode="portfolio"):
    symbol = row['Symbol']
    st.markdown("---")
    
    # 1. Clean Data
    df = df.dropna(how='all')
    if df.empty or ('Close' not in df.columns) or df['Close'].isna().all():
        return

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    if chart_type == "Weekly":
        df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    # 2. Calculate Indicators
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    # Define volume colors
    df['Vol_Color'] = np.where(df['Close'] >= df['Open'], 'rgba(0, 255, 0, 0.5)', 'rgba(255, 0, 0, 0.5)')
    
    plot_df = df[df.index >= start_plot_date]
    if plot_df.empty:
        return
        
    current_price_raw = plot_df['Close'].iloc[-1]
    if isinstance(current_price_raw, (pd.Series, pd.DataFrame)):
        current_price_raw = current_price_raw.iloc[-1]
    current_price = float(current_price_raw)
    
    # 3. Layout Generation
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader(symbol.replace('.NS', ''))
        st.metric("Current Price", f"₹{current_price:.2f}")
        
        if mode == "portfolio":
            target = float(row.get('Target', 0))
            stop_loss = float(row.get('StopLoss', 0))
            purchased_at = float(row.get('PurchasedAt', 0))
            
            if purchased_at > 0:
                pct_return = ((current_price - purchased_at) / purchased_at) * 100
                st.metric("Purchased At", f"₹{purchased_at:.2f}", f"{pct_return:+.1f}% Return")
            if target > 0:
                pct_to_target = ((target - current_price) / current_price) * 100
                st.metric("Target", f"₹{target:.2f}", f"{pct_to_target:.1f}% away")
            if stop_loss > 0:
                pct_to_stop = ((current_price - stop_loss) / current_price) * 100
                st.metric("Stop Loss", f"₹{stop_loss:.2f}", f"-{pct_to_stop:.1f}% risk", delta_color="inverse")
        else:
            # Watchlist Mode
            entry = float(row.get('EntryTrigger', 0))
            notes = str(row.get('Notes', ''))
            if entry > 0:
                pct_to_entry = ((entry - current_price) / current_price) * 100
                st.metric("Entry Trigger", f"₹{entry:.2f}", f"{pct_to_entry:.1f}% to breakout", delta_color="off")
            if notes and notes != 'nan':
                st.info(f"📝 {notes}")

    with col2:
        # Create 3-Row Subplot: Price (60%), Volume (20%), RSI (20%)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        
        # Row 1: Price Action
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], 
                                     low=plot_df['Low'], close=plot_df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['50_MA'], line=dict(color='blue', width=1.5), name='50 MA'), row=1, col=1)
        
        # Add Horizontal Lines & Channels based on mode
        if mode == "portfolio":
            if target > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[target]*len(plot_df), line=dict(color='green', width=2, dash='dash'), name='Target'), row=1, col=1)
            if stop_loss > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[stop_loss]*len(plot_df), line=dict(color='red', width=2, dash='dash'), name='Stop Loss'), row=1, col=1)
            if purchased_at > 0: fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='rgba(0,0,0,0)'), name=f'Purchased @ ₹{purchased_at:.2f}'), row=1, col=1)
        else:
            # Watchlist: Draw straight channel lines based on Google Sheet inputs
            upper_ch = float(row.get('UpperChannel', 0)) if not pd.isna(row.get('UpperChannel', 0)) else 0
            lower_ch = float(row.get('LowerChannel', 0)) if not pd.isna(row.get('LowerChannel', 0)) else 0
            
            if upper_ch > 0 and lower_ch > 0:
                # Creates a beautiful shaded straight channel
                fig.add_hrect(y0=lower_ch, y1=upper_ch, line_width=1.5, fillcolor="gray", opacity=0.1, line_color="gray", row=1, col=1)
            elif upper_ch > 0:
                fig.add_trace(go.Scatter(x=plot_df.index, y=[upper_ch]*len(plot_df), line=dict(color='gray', width=1.5, dash='solid'), name='Upper Channel'), row=1, col=1)
            elif lower_ch > 0:
                fig.add_trace(go.Scatter(x=plot_df.index, y=[lower_ch]*len(plot_df), line=dict(color='gray', width=1.5, dash='solid'), name='Lower Channel'), row=1, col=1)

            if entry > 0:
                fig.add_trace(go.Scatter(x=plot_df.index, y=[entry]*len(plot_df), line=dict(color='purple', width=2, dash='dash'), name='Entry Trigger'), row=1, col=1)

        # Row 2: Volume
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=plot_df['Vol_Color'], name='Volume'), row=2, col=1)
        
        # Row 3: RSI
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='orange', width=1.5), name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=[70]*len(plot_df), line=dict(color='gray', width=1, dash='dash'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=[30]*len(plot_df), line=dict(color='gray', width=1, dash='dash'), showlegend=False), row=3, col=1)
        
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, showlegend=True)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def extract_safe_df(market_data, symbol):
    try:
        if isinstance(market_data.columns, pd.MultiIndex):
            if symbol in market_data.columns.levels[1]: return market_data.xs(symbol, level=1, axis=1).copy()
            elif symbol in market_data.columns.levels[0]: return market_data.xs(symbol, level=0, axis=1).copy()
        else: return market_data.copy()
    except Exception:
        pass
    return pd.DataFrame()

# --- BUILD TABS ---
tab1, tab2 = st.tabs(["💼 Active Portfolio", "🔭 Watchlist Incubator"])

with tab1:
    if not portfolio.empty:
        st.write("#### 🎯 Portfolio Performance")
        summary_data = []
        for index, row in portfolio.iterrows():
            symbol = row['Symbol']
            df = extract_safe_df(market_data, symbol)
            if df.empty or 'Close' not in df.columns: continue
            try:
                purchased_at = float(row.get('PurchasedAt', 0))
                if purchased_at <= 0: continue
                cp_raw = df['Close'].dropna().iloc[-1]
                if isinstance(cp_raw, (pd.Series, pd.DataFrame)): cp_raw = cp_raw.iloc[-1]
                current_price = float(cp_raw)
                pct_change = ((current_price - purchased_at) / purchased_at) * 100
                summary_data.append({"Symbol": symbol.replace('.NS', ''), "Purchased At": purchased_at, "Current Price": current_price, "% Return": pct_change})
            except Exception: pass
            
        if summary_data:
            summary_df = pd.DataFrame(summary_data).sort_values(by="% Return", ascending=False)
            st.dataframe(summary_df, column_config={"% Return": st.column_config.NumberColumn("% Return", format="%.2f %%"), "Purchased At": st.column_config.NumberColumn("Purchased At", format="₹%.2f"), "Current Price": st.column_config.NumberColumn("Current Price", format="₹%.2f")}, use_container_width=True, hide_index=True)
        
        for index, row in portfolio.iterrows():
            df = extract_safe_df(market_data, row['Symbol'])
            render_stock_row(row, df, mode="portfolio")

with tab2:
    if not watchlist.empty:
        st.write("#### 🔭 Radar / Approaching Breakouts")
        watch_summary = []
        for index, row in watchlist.iterrows():
            symbol = row['Symbol']
            df = extract_safe_df(market_data, symbol)
            if df.empty or 'Close' not in df.columns: continue
            try:
                entry = float(row.get('EntryTrigger', 0))
                cp_raw = df['Close'].dropna().iloc[-1]
                if isinstance(cp_raw, (pd.Series, pd.DataFrame)): cp_raw = cp_raw.iloc[-1]
                current_price = float(cp_raw)
                dist = ((entry - current_price) / current_price) * 100 if entry > 0 else 0
                watch_summary.append({"Symbol": symbol.replace('.NS', ''), "Current Price": current_price, "Entry Trigger": entry, "% to Breakout": dist, "Notes": row.get('Notes', '')})
            except Exception: pass

        if watch_summary:
            w_df = pd.DataFrame(watch_summary).sort_values(by="% to Breakout", ascending=True)
            st.dataframe(w_df, column_config={"% to Breakout": st.column_config.NumberColumn("% to Breakout", format="%.2f %%"), "Entry Trigger": st.column_config.NumberColumn("Entry Trigger", format="₹%.2f"), "Current Price": st.column_config.NumberColumn("Current Price", format="₹%.2f")}, use_container_width=True, hide_index=True)

        for index, row in watchlist.iterrows():
            df = extract_safe_df(market_data, row['Symbol'])
            render_stock_row(row, df, mode="watchlist")

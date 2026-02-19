import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 Positional Portfolio Dashboard")

# Replace this string with your published Google Sheet CSV link
SHEET_CSV_URL = "https://doc-08-7s-sheets.googleusercontent.com/pub/1cn5ph2ulkb0qr3lajt32ge2d8/65f02eot0csuce7h325eaf3gn4/1771501330000/118067547509532549602/118067547509532549602/e@2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD?gid=0&single=true&output=csv&dat=AOctwYqNrGNJJwrvof50Q2KpaeEfMEmd9WUPMSXVWHXmoBOmPeJFIL2CYUcT2fWz04VqgVGmm6JNehPXIzPsbXTeireq5abHZKTfaxQTCWaHmmbQoc84L4uWMYhAwXw6ZGMDtMQ-Y9QQz2-o3iVsLzEl9pex87PtUrRQ94ntu1eqwKZojkUJEzyix75Lx9ih3Zof3ujotOfVhKtZcoT1UtgY06Y_oto-4Bq_tTfTKbcKxC3hJldAY2MaFmlMTvG_C3nd22JDX9_JzoT0pknPnGbiyTkVmwIEXfYL_CxoQlg-VMOyh7lzmzQ6_gN9MNP-rF0qJUxC9i59smm3qthtFL7khhrdd8SQfuUeJiQerXB-fI-M_DBwWYEcYp6BI1rd44O85kG-NDId-v2ELHfc4Gy5d-JqN7VTLnmwLYpbiFeji1Vjie8YoxA3w6ESK1TQmm2zCYTMRtXDL0hkqwaZLj7kPemPqZNLcj0kPe9-9fw5spbFIL-Bbznd2Z0bIV7FC2PR9o_A4puYn8QcB-hBYi4Edq4mLLw8b7qWG3jA6NufC8WTTZxkPWsXoQ8cJED8rK7n1ZznKmHm5egUx_WKaYawKe9IVrsKGRVld6DXJbUQZGJ-pP6QT99n-QzniKjq27NbeI_aeLndikjE8kWoaHNliPYujaBONkVg0W_IvUrogNolFnYij7eQRhg8O0r1ZWYpgRbWUUkeZ2dV2JJjvc6gC3eQOmIUchFG68Xd-x4JRcNZeJzmuf6GM6x8dQ58rAo-UR3jVadWSdAp9G_CswddLl90Mfc23kR8TlHL64lcDL1rI9rRd3OV1yKnIx4zjQeEdY-U863ChTXwplg9ldhVBisPELOvDkZplRwvbXB3jiuKUzRd7BX7H8O-dGlJAp-C9TkVbYC_mCrM_sqphXorSpcQrQ5JXPL1gSsdMnGu3qrw6REDZiZ_mwOYAEM6k6MeQsCQR3xxVaqi8pMQ4-o7U8XgnJQT7I7casdmzRDQz1d65jGmGYNXLGLPdRw07MHpA-b4J-KNcjfX_1EaX5Q65P-SGQ"

@st.cache_data(ttl=3600) # Caches data for 1 hour so it loads fast
def load_portfolio():
    df = pd.read_csv(SHEET_CSV_URL)
    return df

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, days=100):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days + 50) # Extra days to calculate 50-DMA
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return data

# --- LOAD DATA ---
try:
    portfolio = load_portfolio()
except Exception as e:
    st.error("Error loading Google Sheet. Please check the link.")
    st.stop()

# --- DASHBOARD UI ---
st.write("### Market Overview")
days_to_plot = st.slider("Select chart timeframe (Days)", min_value=30, max_value=200, value=100)

# Loop through each stock in your Google Sheet
for index, row in portfolio.iterrows():
    symbol = row['Symbol']
    target = row['Target']
    stop_loss = row['StopLoss']
    
    st.markdown(f"---")
    
    # Fetch data
    df = fetch_stock_data(symbol)
    
    if df.empty:
        st.warning(f"No data found for {symbol}. Check the symbol name.")
        continue

    # Calculate 50-Day Moving Average
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Slice dataframe to the user's selected timeframe
    plot_df = df.tail(days_to_plot)
    current_price = plot_df['Close'].iloc[-1]
    
    # Calculate percentages
    pct_to_target = ((target - current_price) / current_price) * 100
    pct_to_stop = ((current_price - stop_loss) / current_price) * 100

    # Layout: Metrics on the left, Chart on the right (stack vertically on mobile)
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
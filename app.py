import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import itertools

# --- CONFIGURATION ---
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📈 Positional Trading Dashboard")

# ⚠️ REPLACE THESE TWO STRINGS WITH YOUR GOOGLE SHEET CSV LINKS
PORTFOLIO_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD/pub?gid=0&single=true&output=csv"
WATCHLIST_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT5msMoKIvOtgoNeVJb41T2pRasfeAMwou0U_bz_4vqS_AzNIK_iHL88Z0OTN4za2_7RGO58S-jfCbD/pub?gid=186620296&single=true&output=csv"

@st.cache_data(ttl=300)
def load_csv(url):
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_all_stock_data(symbols):
    data = yf.download(symbols, period="5y", threads=True, progress=False)
    return data

def get_safe_price(series):
    clean_series = series.dropna()
    if not clean_series.empty:
        val = clean_series.iloc[-1]
        if isinstance(val, (pd.Series, pd.DataFrame)):
            return float(val.iloc[-1])
        return float(val)
    return 0.0

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==============================================================================
# LUXALGO SUPERTREND AI - DYNAMIC OPTIMIZATION ENGINE
# ==============================================================================

def calculate_rma(data, length):
    rma = np.zeros_like(data)
    if len(data) <= length:
        return rma
    rma[length] = np.mean(data[1:length+1])
    for i in range(length + 1, len(data)):
        rma[i] = (rma[i-1] * (length - 1) + data[i]) / length
    return rma

def calculate_atr(high, low, close, length):
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(hl, np.maximum(hc, lc))
    tr = np.insert(tr, 0, high[0] - low[0])
    return calculate_rma(tr, length)

def simulate_supertrend_ai(high, low, close, atr, atr_length, min_mult=1.0, max_mult=5.0, step=0.5, perf_alpha=10, target_cluster='Best'):
    factors = np.arange(min_mult, max_mult + step, step)
    n_factors = len(factors)
    n_bars = len(close)
    hl2 = (high + low) / 2.0

    uppers = np.zeros((n_factors, n_bars))
    lowers = np.zeros((n_factors, n_bars))
    trends = np.ones((n_factors, n_bars))
    outputs = np.zeros((n_factors, n_bars))
    perfs = np.zeros((n_factors, n_bars))

    dyn_upper = np.zeros(n_bars)
    dyn_lower = np.zeros(n_bars)
    dyn_trend = np.zeros(n_bars)

    alpha = 2.0 / (perf_alpha + 1)

    for f_idx, factor in enumerate(factors):
        uppers[f_idx, 0] = hl2[0] + atr[0] * factor
        lowers[f_idx, 0] = hl2[0] - atr[0] * factor
        outputs[f_idx, 0] = lowers[f_idx, 0]

    dyn_upper[0] = hl2[0] + atr[0] * np.mean(factors)
    dyn_lower[0] = hl2[0] - atr[0] * np.mean(factors)
    dyn_trend[0] = 1

    for i in range(1, n_bars):
        current_perfs = np.zeros(n_factors)

        for f_idx, factor in enumerate(factors):
            up = hl2[i] + atr[i] * factor
            dn = hl2[i] - atr[i] * factor

            uppers[f_idx, i] = min(up, uppers[f_idx, i-1]) if close[i-1] < uppers[f_idx, i-1] else up
            lowers[f_idx, i] = max(dn, lowers[f_idx, i-1]) if close[i-1] > lowers[f_idx, i-1] else dn

            if close[i] > uppers[f_idx, i]:
                trends[f_idx, i] = 1
            elif close[i] < lowers[f_idx, i]:
                trends[f_idx, i] = 0
            else:
                trends[f_idx, i] = trends[f_idx, i-1]

            outputs[f_idx, i] = lowers[f_idx, i] if trends[f_idx, i] == 1 else uppers[f_idx, i]

            diff = np.sign(close[i-1] - outputs[f_idx, i-1]) if i > 1 else 0
            raw_perf = (close[i] - close[i-1]) * diff
            perfs[f_idx, i] = perfs[f_idx, i-1] + alpha * (raw_perf - perfs[f_idx, i-1])
            current_perfs[f_idx] = perfs[f_idx, i]

        target_factor = np.mean(factors)
        if i > atr_length:
            centroids = np.percentile(current_perfs, [25, 50, 75])
            for _ in range(10):
                dists = np.abs(current_perfs[:, None] - centroids)
                labels = np.argmin(dists, axis=1)
                new_centroids = np.zeros(3)
                for c_idx in range(3):
                    if np.sum(labels == c_idx) > 0:
                        new_centroids[c_idx] = np.mean(current_perfs[labels == c_idx])
                    else:
                        new_centroids[c_idx] = centroids[c_idx]
                if np.all(new_centroids == centroids):
                    break
                centroids = new_centroids

            sorted_indices = np.argsort(centroids)
            if target_cluster == 'Best':
                cluster_label = sorted_indices[2]
            elif target_cluster == 'Average':
                cluster_label = sorted_indices[1]
            else:
                cluster_label = sorted_indices[0]

            target_factors = factors[labels == cluster_label]
            if len(target_factors) > 0:
                target_factor = np.mean(target_factors)

        dyn_up = hl2[i] + atr[i] * target_factor
        dyn_dn = hl2[i] - atr[i] * target_factor

        dyn_upper[i] = min(dyn_up, dyn_upper[i-1]) if close[i-1] < dyn_upper[i-1] else dyn_up
        dyn_lower[i] = max(dyn_dn, dyn_lower[i-1]) if close[i-1] > dyn_lower[i-1] else dyn_dn

        if close[i] > dyn_upper[i]:
            dyn_trend[i] = 1
        elif close[i] < dyn_lower[i]:
            dyn_trend[i] = 0
        else:
            dyn_trend[i] = dyn_trend[i-1]

    st_line = np.where(dyn_trend == 1, dyn_lower, dyn_upper)
    return st_line, dyn_trend

def backtest_signals(df, trend_signals):
    df_bt = df.copy().reset_index(drop=True)
    df_bt['Trend'] = trend_signals
    df_bt['Signal'] = df_bt['Trend'].diff()

    entries = df_bt.index[df_bt['Signal'] == 1].tolist()
    exits = df_bt.index[df_bt['Signal'] == -1].tolist()
    trades = []

    for entry_idx in entries:
        valid_exits = [x for x in exits if x > entry_idx]
        exit_idx = valid_exits[0] if valid_exits else len(df_bt) - 1

        trade_entry_idx = min(entry_idx + 1, len(df_bt) - 1)
        trade_exit_idx = min(exit_idx + 1, len(df_bt) - 1)

        if trade_entry_idx < len(df_bt) and trade_exit_idx < len(df_bt) and trade_entry_idx != trade_exit_idx:
            buy_price = df_bt.loc[trade_entry_idx, 'Open']
            sell_price = df_bt.loc[trade_exit_idx, 'Open']
            if buy_price > 0:
                profit_pct = (sell_price - buy_price) / buy_price
                trades.append(profit_pct)

    win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0
    comp_roi = (np.prod([1 + t for t in trades]) - 1) * 100 if trades else 0
    return len(trades), win_rate, comp_roi

def get_optimized_supertrend(df):
    """Grid search across parameters to find the highest backtested ROI for this stock."""
    high = np.squeeze(df['High'].values)
    low = np.squeeze(df['Low'].values)
    close = np.squeeze(df['Close'].values)

    atr_lengths = [10, 14, 21]
    perf_alphas = [10, 20]
    target_clusters = ['Best', 'Average']
    min_mult, max_mult, step = 1.0, 5.0, 0.5

    best_roi = -99999
    best_params = {}
    best_st_line = None
    best_trend = None

    combinations = list(itertools.product(atr_lengths, perf_alphas, target_clusters))

    for atr_len, p_alpha, cluster in combinations:
        atr = calculate_atr(high, low, close, atr_len)
        st_line, trend = simulate_supertrend_ai(
            high, low, close, atr,
            atr_length=atr_len,
            min_mult=min_mult, max_mult=max_mult, step=step,
            perf_alpha=p_alpha, target_cluster=cluster
        )
        n_trades, win_rate, roi = backtest_signals(df, trend)

        if roi > best_roi:
            best_roi = roi
            best_params = {
                'atr_len': atr_len,
                'perf_alpha': p_alpha,
                'cluster': cluster,
                'trades': n_trades,
                'win_rate': round(win_rate * 100, 1),
                'roi': round(roi, 1)
            }
            best_st_line = st_line
            best_trend = trend

    return pd.Series(best_st_line, index=df.index), pd.Series(best_trend, index=df.index), best_params

# ==============================================================================
# DATA LOADING & INITIALIZATION
# ==============================================================================

portfolio = load_csv(PORTFOLIO_CSV_URL)
watchlist = load_csv(WATCHLIST_CSV_URL)

port_symbols = portfolio['Symbol'].dropna().unique().tolist() if not portfolio.empty else []
watch_symbols = watchlist['Symbol'].dropna().unique().tolist() if not watchlist.empty else []
all_symbols = list(set(port_symbols + watch_symbols))

if not all_symbols:
    st.error("No symbols found. Please check your Google Sheet CSV links.")
    st.stop()

col_t1, col_t2 = st.columns(2)
with col_t1:
    days_to_plot = st.slider("Select chart visual timeframe (Days)", min_value=30, max_value=1825, value=1095)
with col_t2:
    chart_type = st.radio("Chart Timeframe", ["Daily", "Weekly"], horizontal=True)

with st.spinner('Fetching market data & running SuperTrend AI optimizations...'):
    market_data = fetch_all_stock_data(all_symbols)

start_plot_date = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=days_to_plot))

# ==============================================================================
# HELPER FUNCTIONS & RENDERER
# ==============================================================================

def extract_safe_df(market_data, symbol):
    try:
        if isinstance(market_data.columns, pd.MultiIndex):
            if symbol in market_data.columns.levels[1]: return market_data.xs(symbol, level=1, axis=1).copy()
            elif symbol in market_data.columns.levels[0]: return market_data.xs(symbol, level=0, axis=1).copy()
        else: return market_data.copy()
    except Exception: pass
    return pd.DataFrame()

def render_stock_row(row, df, mode="portfolio"):
    symbol = row['Symbol']
    st.markdown("---")
    
    df = df.dropna(how='all')
    if df.empty or ('Close' not in df.columns) or df['Close'].dropna().empty: return

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if isinstance(df[col], pd.DataFrame): df[col] = df[col].iloc[:, 0]

    if chart_type == "Weekly":
        df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    # Run Grid-Search Optimizer for LuxAlgo AI SuperTrend on BOTH tabs
    st_line, st_trend, opt_params = get_optimized_supertrend(df)
    df['ST'] = st_line
    df['ST_Trend'] = st_trend

    # TradingView Exact Colors
    tv_bg = "#131722"
    tv_grid = "#2B2B36"
    tv_text = "#B2B5BE"
    bull_color = "#26A69A"
    bear_color = "#EF5350"
    
    df['Vol_Color'] = np.where(df['Close'] >= df['Open'], 'rgba(38, 166, 154, 0.5)', 'rgba(239, 83, 80, 0.5)')
    
    plot_df = df[df.index >= start_plot_date]
    if plot_df.empty: return
        
    current_price = get_safe_price(plot_df['Close'])
    if current_price == 0: return

    col1, col2, col3 = st.columns([1.5, 4.5, 2])
    
    with col1:
        st.subheader(symbol.replace('.NS', ''))
        st.metric("Current Price", f"₹{current_price:.2f}")
        
        if mode == "portfolio":
            target = float(row.get('Target', 0)) if not pd.isna(row.get('Target', 0)) else 0
            stop_loss = float(row.get('StopLoss', 0)) if not pd.isna(row.get('StopLoss', 0)) else 0
            purchased_at = float(row.get('PurchasedAt', 0)) if not pd.isna(row.get('PurchasedAt', 0)) else 0
            
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
            entry = float(row.get('EntryTrigger', 0)) if not pd.isna(row.get('EntryTrigger', 0)) else 0
            notes = str(row.get('Notes', ''))
            if entry > 0:
                pct_to_entry = ((entry - current_price) / current_price) * 100
                st.metric("Entry Trigger", f"₹{entry:.2f}", f"{pct_to_entry:.1f}% to breakout", delta_color="off")
            if notes and notes != 'nan':
                st.info(f"📝 {notes}")

        # Display AI Optimization Parameter Badge
        if opt_params:
            st.caption(f"🤖 **Optimized SuperTrend AI Parameters**\n"
                       f"- ATR Length: `{opt_params['atr_len']}` | Alpha: `{opt_params['perf_alpha']}`\n"
                       f"- Cluster: `{opt_params['cluster']}`\n"
                       f"- Backtested Win Rate: `{opt_params['win_rate']}%`\n"
                       f"- Backtested ROI: `{opt_params['roi']:+.1f}%`")

    with col2:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.6, 0.2, 0.2])
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'],
            name='Price',
            increasing_line_color=bull_color, increasing_fillcolor=bull_color,
            decreasing_line_color=bear_color, decreasing_fillcolor=bear_color
        ), row=1, col=1)
        
        # SuperTrend AI Lines (Plotted for BOTH Portfolio and Watchlist)
        st_green = np.where(plot_df['ST_Trend'] == 1, plot_df['ST'], np.nan)
        st_red = np.where(plot_df['ST_Trend'] == -1, plot_df['ST'], np.nan)
        fig.add_trace(go.Scatter(x=plot_df.index, y=st_green, line=dict(color=bull_color, width=2), name='SuperTrend (Bull)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=st_red, line=dict(color=bear_color, width=2), name='SuperTrend (Bear)'), row=1, col=1)
        
        # Buy / Sell Signals (Triangles)
        trend_diff = plot_df['ST_Trend'].diff()
        buy_sigs = plot_df[trend_diff == 1]
        sell_sigs = plot_df[trend_diff == -1]
        
        fig.add_trace(go.Scatter(x=buy_sigs.index, y=buy_sigs['Low']*0.95, mode='markers', marker=dict(symbol='triangle-up', color=bull_color, size=12), name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_sigs.index, y=sell_sigs['High']*1.05, mode='markers', marker=dict(symbol='triangle-down', color=bear_color, size=12), name='Sell Signal'), row=1, col=1)

        if mode == "portfolio":
            if target > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[target]*len(plot_df), line=dict(color=bull_color, width=1.5, dash='dash'), name='Target'), row=1, col=1)
            if stop_loss > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[stop_loss]*len(plot_df), line=dict(color=bear_color, width=1.5, dash='dash'), name='Stop Loss'), row=1, col=1)
            if purchased_at > 0: fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='rgba(0,0,0,0)'), name=f'Purchased @ ₹{purchased_at:.2f}'), row=1, col=1)
        else:
            upper_ch = float(row.get('UpperChannel', 0)) if not pd.isna(row.get('UpperChannel', 0)) else 0
            lower_ch = float(row.get('LowerChannel', 0)) if not pd.isna(row.get('LowerChannel', 0)) else 0
            
            if upper_ch > 0 and lower_ch > 0:
                fig.add_hrect(y0=lower_ch, y1=upper_ch, line_width=1.5, fillcolor="#FFD600", opacity=0.05, line_color="#FFD600", row=1, col=1)
            elif upper_ch > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[upper_ch]*len(plot_df), line=dict(color='#FFD600', width=1.5, dash='solid'), name='Upper Channel'), row=1, col=1)
            elif lower_ch > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[lower_ch]*len(plot_df), line=dict(color='#FFD600', width=1.5, dash='solid'), name='Lower Channel'), row=1, col=1)
            if entry > 0: fig.add_trace(go.Scatter(x=plot_df.index, y=[entry]*len(plot_df), line=dict(color='#E040FB', width=1.5, dash='dash'), name='Entry Trigger'), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=plot_df['Vol_Color'], name='Volume'), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#7E57C2', width=1.5), name='RSI'), row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="#7E57C2", opacity=0.1, line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=[70]*len(plot_df), line=dict(color=tv_grid, width=1, dash='dash'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=[30]*len(plot_df), line=dict(color=tv_grid, width=1, dash='dash'), showlegend=False), row=3, col=1)
        
        # TRADINGVIEW STYLING
        fig.update_layout(
            height=600, 
            margin=dict(l=0, r=0, t=10, b=0), 
            xaxis_rangeslider_visible=False, 
            showlegend=False,
            plot_bgcolor=tv_bg,
            paper_bgcolor=tv_bg,
            font=dict(color=tv_text, size=10),
            dragmode='pan',
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=tv_grid, zeroline=False, showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dot', spikecolor=tv_text)
        fig.update_yaxes(side='right', showgrid=True, gridwidth=1, gridcolor=tv_grid, zeroline=False, tickfont=dict(color=tv_text))
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

    with col3:
        st.markdown("##### 📰 Latest News")
        with st.container(height=550):
            try:
                ticker = yf.Ticker(symbol)
                news_items = ticker.news
                
                if news_items:
                    valid_articles = 0
                    for article in news_items:
                        if valid_articles >= 5: 
                            break
                            
                        title = "No Title"
                        link = "#"
                        pub_date = None
                        
                        if 'content' in article:
                            content = article['content']
                            title = content.get('title', 'No Title')
                            link = content.get('canonicalUrl', {}).get('url', '#')
                            pub_date = content.get('pubDate') or content.get('providerPublishTime')
                        else:
                            title = article.get('title', 'No Title')
                            link = article.get('link', '#')
                            pub_date = article.get('providerPublishTime')
                            
                        date_label = ""
                        if pub_date:
                            try:
                                if isinstance(pub_date, (int, float)):
                                    dt = datetime.datetime.fromtimestamp(pub_date)
                                else:
                                    dt = pd.to_datetime(pub_date)
                                date_label = f"`{dt.strftime('%d %b')}` "
                            except Exception:
                                date_label = ""

                        if title and title != 'No Title':
                            st.markdown(f"- {date_label}[{title}]({link})")
                            st.divider()
                            valid_articles += 1
                            
                    if valid_articles == 0:
                        st.write("No recent news found.")
                else:
                    st.write("No recent news found.")
            except Exception:
                st.write("Unable to load news at this time.")

# ==============================================================================
# TABS CONSTRUCTION
# ==============================================================================

tab1, tab2 = st.tabs(["💼 Active Portfolio", "🔭 Watchlist Incubator"])

with tab1:
    if not portfolio.empty:
        st.write("#### 🎯 Portfolio Performance")
        summary_data = []
        for index, row in portfolio.iterrows():
            symbol = row['Symbol']
            df = extract_safe_df(market_data, symbol)
            if df.empty: continue
            
            purchased_at = float(row.get('PurchasedAt', 0)) if not pd.isna(row.get('PurchasedAt', 0)) else 0
            if purchased_at <= 0: continue
            
            current_price = get_safe_price(df['Close'])
            if current_price == 0: continue
                
            pct_change = ((current_price - purchased_at) / purchased_at) * 100
            summary_data.append({"Symbol": symbol.replace('.NS', ''), "Purchased At": purchased_at, "Current Price": current_price, "% Return": pct_change})
            
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
            if df.empty: continue
            
            entry = float(row.get('EntryTrigger', 0)) if not pd.isna(row.get('EntryTrigger', 0)) else 0
            current_price = get_safe_price(df['Close'])
            if current_price == 0: continue
                
            dist = ((entry - current_price) / current_price) * 100 if entry > 0 else 0
            watch_summary.append({"Symbol": symbol.replace('.NS', ''), "Current Price": current_price, "Entry Trigger": entry, "% to Breakout": dist, "Notes": str(row.get('Notes', ''))})

        if watch_summary:
            w_df = pd.DataFrame(watch_summary).sort_values(by="% to Breakout", ascending=True)
            st.dataframe(w_df, column_config={"% to Breakout": st.column_config.NumberColumn("% to Breakout", format="%.2f %%"), "Entry Trigger": st.column_config.NumberColumn("Entry Trigger", format="₹%.2f"), "Current Price": st.column_config.NumberColumn("Current Price", format="₹%.2f")}, use_container_width=True, hide_index=True)

        for index, row in watchlist.iterrows():
            df = extract_safe_df(market_data, row['Symbol'])
            render_stock_row(row, df, mode="watchlist")

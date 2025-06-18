import streamlit as st
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, date
import time
import matplotlib.pyplot as plt
import numpy as np
import nltk
import os # For potential local file caching (optional future step)
import json # For potential local file caching (optional future step)

# --- Page Configuration ---
st.set_page_config(page_title="Nifty Sentiment Backtester", layout="wide")

# --- Global Initialization (done once) ---
@st.cache_resource 
def get_sentiment_analyzer():
    try:
        analyzer = SentimentIntensityAnalyzer()
        analyzer.polarity_scores("test") 
        return analyzer
    except LookupError:
        st.info("VADER lexicon not found. Attempting to download...")
        try:
            nltk.download('vader_lexicon', quiet=True)
            analyzer = SentimentIntensityAnalyzer()
            st.success("VADER lexicon downloaded successfully.")
            return analyzer
        except Exception as e:
            st.error(f"Failed to download VADER lexicon: {e}.")
            return None
    except Exception as e:
        st.error(f"Error initializing sentiment analyzer: {e}")
        return None
analyzer = get_sentiment_analyzer()

# --- Cached NewsAPI Client ---
@st.cache_resource # Cache the client object itself
def get_newsapi_client(api_key):
    st.write(f"Initializing NewsAPI Client (API Key ending: ...{api_key[-4:] if api_key and len(api_key) > 4 else '****'})...")
    if not api_key or api_key == "YOUR_NEWS_API_KEY":
        st.error("NewsAPI Key is invalid or placeholder. Please provide a valid key.")
        return None
    try:
        client = NewsApiClient(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize NewsAPI client: {e}")
        return None

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600, show_spinner=False) 
def get_news_for_day(_newsapi_client_instance, date_obj_for_news): # Use precise names for clarity
    # Note: _newsapi_client_instance MUST be the cached client for this function's cache to be effective
    if _newsapi_client_instance is None:
        st.warning(f"NewsAPI client not available. Skipping news fetch for {date_obj_for_news.strftime('%Y-%m-%d')}.")
        return []

    date_str = date_obj_for_news.strftime('%Y-%m-%d')
    query_terms = '"Nifty 50" OR "Indian economy" OR "Sensex" OR "Indian stock market" OR "NSE India" OR "BSE India"'
    articles_data = []
    # print(f"Fetching news for: {date_str}...") # Console log
    try:
        all_articles = _newsapi_client_instance.get_everything(
            q=query_terms, from_param=date_str, to=date_str, language='en', sort_by='relevancy', page_size=100
        )
        if all_articles['status'] == 'ok':
            for article in all_articles['articles']:
                content = (article['title'] or "") + " " + (article['description'] or "")
                if content.strip(): articles_data.append({'content': content.strip()})
        else:
            msg = all_articles.get('message', 'Unknown NewsAPI error')
            code = all_articles.get('code', 'N/A')
            # st.caption(f"News API for {date_str}: {msg} (Code: {code})") # Less intrusive
            if code == 'rateLimited': 
                st.toast(f"Rate limited by NewsAPI for {date_str}. Pausing...", icon="⏳")
                time.sleep(65) # This pause will block Streamlit's thread for this user
            elif code == 'parameterInvalid' and 'far in the past' in msg:
                 st.caption(f"News for {date_str} outside allowed window.") # Inform user
    except Exception as e:
        st.caption(f"News fetch exception for {date_str}: {str(e)[:100]}...")
    time.sleep(1.2) # API courtesy delay
    return articles_data

def get_sentiment_score(text_content):
    if not analyzer: return 0.0
    return analyzer.polarity_scores(text_content)['compound'] if text_content and isinstance(text_content, str) else 0.0

@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def get_market_data(ticker, fetch_start_date, fetch_end_date):
    start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_str_yf = (fetch_end_date + timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_str, end=end_str_yf, interval="1d", auto_adjust=True, progress=False)
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        data = data[data.index.normalize() <= pd.to_datetime(fetch_end_date).normalize()]
    return data

# --- MAIN BACKTESTING FUNCTION (WITH ROLLING SENTIMENT) ---
def run_sentiment_backtest(newsapi_key_from_ui, nifty_ticker_input, end_date_input_dt,
                           days_backtest_input, initial_capital_input, positive_thresh_input,
                           negative_thresh_input, transaction_cost_pct_input,
                           sentiment_lookback_days_param):
    if not analyzer:
        st.error("Sentiment analyzer not initialized.")
        return None, {}
    
    # Get the (potentially cached) NewsAPI client
    newsapi_client_cached = get_newsapi_client(newsapi_key_from_ui)
    if not newsapi_client_cached:
        st.error("NewsAPI client could not be initialized. Check API Key and console.")
        return None, {}

    yfinance_fetch_start_date = end_date_input_dt - timedelta(days=days_backtest_input + 15 + sentiment_lookback_days_param)
    yfinance_fetch_end_date = end_date_input_dt
    nifty_data_fetched = get_market_data(nifty_ticker_input, yfinance_fetch_start_date, yfinance_fetch_end_date)

    if nifty_data_fetched.empty:
        st.error("No Nifty data fetched."); return None, {}
    if len(nifty_data_fetched) < days_backtest_input:
        st.warning(f"Insufficient Nifty data: Fetched {len(nifty_data_fetched)}, need {days_backtest_input}. Using all.")
        nifty_data = nifty_data_fetched.copy()
    else:
        nifty_data = nifty_data_fetched.tail(days_backtest_input).copy()
    if nifty_data.empty:
        st.error("Nifty data empty after filtering."); return None, {}

    actual_backtest_start_dt = nifty_data.index[0].to_pydatetime()
    actual_backtest_end_dt = nifty_data.index[-1].to_pydatetime()
    st.info(f"Actual backtest: {actual_backtest_start_dt.strftime('%Y-%m-%d')} to {actual_backtest_end_dt.strftime('%Y-%m-%d')} ({len(nifty_data)} days). Sentiment Lookback: {sentiment_lookback_days_param} days.")

    for col in ['Sentiment', 'Rolling_Sentiment', 'Signal', 'Position_State', 'Holdings_Units', 'Cash',
                'Portfolio_Value', 'Strategy_Return_Daily', 'Trades_Action']:
        nifty_data[col] = 0.0 if col not in ['Signal', 'Position_State', 'Trades_Action'] else 0

    current_cash = initial_capital_input; current_holdings_units = 0.0; current_position_state = 0
    daily_sentiments_history = []
    progress_bar = st.progress(0); status_text = st.empty()

    if sentiment_lookback_days_param > 1:
        # st.write("Pre-calculating initial sentiment history...") # Can be verbose
        for k_lookback in range(sentiment_lookback_days_param - 1, 0, -1):
            hist_news_date = actual_backtest_start_dt - timedelta(days=k_lookback)
            hist_articles = get_news_for_day(newsapi_client_cached, hist_news_date) # Use cached client
            hist_sentiment = np.mean([get_sentiment_score(article['content']) for article in hist_articles]) if hist_articles else 0.0
            daily_sentiments_history.append(hist_sentiment)
        # st.write(f"Pre-calculated {len(daily_sentiments_history)} days of sentiment history.")

    for i in range(len(nifty_data)):
        progress_bar.progress((i + 1) / len(nifty_data)); idx = nifty_data.index[i]
        status_text.text(f"Processing Day {i+1}/{len(nifty_data)}: {idx.strftime('%Y-%m-%d')}")
        nifty_data.loc[idx, 'Cash'] = current_cash; nifty_data.loc[idx, 'Holdings_Units'] = current_holdings_units
        nifty_data.loc[idx, 'Position_State'] = current_position_state

        open_price_candidate = nifty_data['Open'].iloc[i]
        current_open_price = open_price_candidate.item() if isinstance(open_price_candidate, pd.Series) and not open_price_candidate.empty else open_price_candidate
        is_open_price_scalar_valid = pd.notna(current_open_price) and current_open_price > 0
        pre_trade_portfolio_value = current_cash + (current_holdings_units * current_open_price) if is_open_price_scalar_valid else (nifty_data['Portfolio_Value'].iloc[i-1] if i > 0 else initial_capital_input)
        nifty_data.loc[idx, 'Portfolio_Value'] = pre_trade_portfolio_value

        news_date_for_current_signal = idx.to_pydatetime() - timedelta(days=1)
        articles = get_news_for_day(newsapi_client_cached, news_date_for_current_signal) # Use cached client
        todays_raw_sentiment = np.mean([get_sentiment_score(article['content']) for article in articles]) if articles else 0.0
        nifty_data.loc[idx, 'Sentiment'] = todays_raw_sentiment
        daily_sentiments_history.append(todays_raw_sentiment)
        if len(daily_sentiments_history) > sentiment_lookback_days_param: daily_sentiments_history.pop(0)
        final_sentiment_for_signal = np.mean(daily_sentiments_history) if len(daily_sentiments_history) == sentiment_lookback_days_param else (daily_sentiments_history[-1] if sentiment_lookback_days_param == 1 and daily_sentiments_history else 0.0)
        nifty_data.loc[idx, 'Rolling_Sentiment'] = final_sentiment_for_signal
        
        signal = 0
        if final_sentiment_for_signal > positive_thresh_input: signal = 1
        elif final_sentiment_for_signal < negative_thresh_input: signal = -1
        nifty_data.loc[idx, 'Signal'] = signal

        if not is_open_price_scalar_valid: nifty_data.loc[idx, 'Trades_Action'] = 0
        else: # Trade Execution
            if current_position_state == 0 and signal == 1 and current_cash > 0:
                value_to_invest = current_cash / (1 + transaction_cost_pct_input)
                units_to_buy = value_to_invest / current_open_price 
                cost = units_to_buy * current_open_price; fee = cost * transaction_cost_pct_input
                current_holdings_units = units_to_buy; current_cash -= (cost + fee)
                if current_cash < 0: current_cash = 0
                current_position_state = 1; nifty_data.loc[idx, 'Trades_Action'] = 1
            elif current_position_state == 1 and signal == -1 and current_holdings_units > 0:
                proceeds = current_holdings_units * current_open_price 
                fee = proceeds * transaction_cost_pct_input
                current_cash += (proceeds - fee)
                current_holdings_units = 0.0; current_position_state = 0
                nifty_data.loc[idx, 'Trades_Action'] = -1
            else: nifty_data.loc[idx, 'Trades_Action'] = 0
        
        nifty_data.loc[idx, 'Cash'] = current_cash; nifty_data.loc[idx, 'Holdings_Units'] = current_holdings_units
        nifty_data.loc[idx, 'Position_State'] = current_position_state

        close_price_candidate = nifty_data['Close'].iloc[i]
        current_close_price = close_price_candidate.item() if isinstance(close_price_candidate, pd.Series) and not close_price_candidate.empty else close_price_candidate
        is_close_price_scalar_valid = pd.notna(current_close_price) and current_close_price > 0
        eod_portfolio_value = current_cash + (current_holdings_units * current_close_price) if is_close_price_scalar_valid else nifty_data.loc[idx, 'Portfolio_Value']
        nifty_data.loc[idx, 'Portfolio_Value'] = eod_portfolio_value

        if i > 0: # Daily Return
            prev_eod_val = nifty_data['Portfolio_Value'].iloc[i-1]
            if pd.notna(prev_eod_val) and prev_eod_val != 0 and pd.notna(eod_portfolio_value):
                nifty_data.loc[idx, 'Strategy_Return_Daily'] = (eod_portfolio_value / prev_eod_val) - 1
        else:
            if initial_capital_input != 0 and pd.notna(eod_portfolio_value):
                nifty_data.loc[idx, 'Strategy_Return_Daily'] = (eod_portfolio_value / initial_capital_input) - 1
    
    status_text.text("Backtest complete!"); progress_bar.empty()
    results = {} # Performance Metrics (same as before)
    # ... (The rest of your performance metrics calculation, it was already correct) ...
    if nifty_data.empty or 'Portfolio_Value' not in nifty_data.columns or nifty_data['Portfolio_Value'].empty:
        final_val = initial_capital_input
    else:
        final_val = nifty_data['Portfolio_Value'].ffill().iloc[-1]
    results['initial_capital'] = initial_capital_input
    results['final_portfolio_value'] = final_val
    results['total_strategy_return_pct'] = ((final_val / initial_capital_input) - 1) * 100 if initial_capital_input != 0 else 0.0
    buy_hold_calculated_flag = False
    if not nifty_data.empty and 'Open' in nifty_data.columns and 'Close' in nifty_data.columns and \
       len(nifty_data['Open']) > 0 and len(nifty_data['Close']) > 0: 
        open_price_bh_candidate = nifty_data['Open'].iloc[0]
        close_price_bh_candidate = nifty_data['Close'].iloc[-1]
        nifty_initial_price_for_bh = open_price_bh_candidate.item() if isinstance(open_price_bh_candidate, pd.Series) and not open_price_bh_candidate.empty else open_price_bh_candidate
        nifty_final_price_for_bh = close_price_bh_candidate.item() if isinstance(close_price_bh_candidate, pd.Series) and not close_price_bh_candidate.empty else close_price_bh_candidate
        if pd.notna(nifty_initial_price_for_bh) and nifty_initial_price_for_bh > 0 and \
           pd.notna(nifty_final_price_for_bh) and nifty_final_price_for_bh > 0:
            results['nifty_buy_hold_return_pct'] = ((nifty_final_price_for_bh / nifty_initial_price_for_bh) - 1) * 100
            buy_hold_calculated_flag = True
    if not buy_hold_calculated_flag: results['nifty_buy_hold_return_pct'] = "N/A"
    results['num_trades'] = nifty_data[nifty_data['Trades_Action'] != 0]['Trades_Action'].count() if not nifty_data.empty and 'Trades_Action' in nifty_data.columns else 0
    valid_returns = nifty_data['Strategy_Return_Daily'].dropna() if not nifty_data.empty and 'Strategy_Return_Daily' in nifty_data.columns else pd.Series(dtype=float)
    if len(valid_returns) > 1:
        mean_daily_return = valid_returns.mean()
        std_daily_return = valid_returns.std()
        if pd.notna(std_daily_return) and std_daily_return != 0:
            results['sharpe_ratio_annualized'] = (mean_daily_return / std_daily_return) * np.sqrt(252)
        else: results['sharpe_ratio_annualized'] = "N/A"
    else: results['sharpe_ratio_annualized'] = "N/A"
    results['actual_start_date'] = actual_backtest_start_dt if not nifty_data.empty else pd.NaT
    results['actual_end_date'] = actual_backtest_end_dt if not nifty_data.empty else pd.NaT
    return nifty_data, results

# --- PLOTTING FUNCTION (already updated for Rolling_Sentiment) ---
# (Plotting function from previous response is fine, no changes needed here if it already takes sentiment_lookback_days_param)
def plot_backtest_results(nifty_data_df, results_dict, initial_capital_val, positive_thresh_val, negative_thresh_val, sentiment_lookback_days_param): # Added param
    if nifty_data_df is None or nifty_data_df.empty:
        st.warning("No data to plot.")
        return None
    fig = plt.figure(figsize=(16, 13));
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except: plt.style.use('seaborn-darkgrid')

    ax1 = plt.subplot(3, 1, 1) 
    if 'Portfolio_Value' in nifty_data_df.columns and not nifty_data_df['Portfolio_Value'].empty:
        nifty_data_df['Portfolio_Normalized'] = nifty_data_df['Portfolio_Value'] / initial_capital_val if initial_capital_val != 0 else 0.0
        nifty_data_df['Portfolio_Normalized'].plot(ax=ax1, label='Strategy Portfolio Value', color='royalblue')
    first_close_price_scalar = np.nan 
    if 'Close' in nifty_data_df.columns and not nifty_data_df['Close'].empty:
        first_close_candidate = nifty_data_df['Close'].iloc[0]
        first_close_price_scalar = first_close_candidate.item() if isinstance(first_close_candidate, pd.Series) and not first_close_candidate.empty else first_close_candidate
        if pd.notna(first_close_price_scalar) and first_close_price_scalar != 0:
            nifty_data_df['Nifty_BH_Normalized'] = nifty_data_df['Close'] / first_close_price_scalar
            nifty_data_df['Nifty_BH_Normalized'].plot(ax=ax1, label='Nifty 50 (B&H from Day 1 Close)', color='grey', linestyle='--')
    title_start_date = results_dict.get('actual_start_date'); title_end_date = results_dict.get('actual_end_date')
    title_start_str = title_start_date.strftime("%Y-%m-%d") if pd.notna(title_start_date) else "N/A"
    title_end_str = title_end_date.strftime("%Y-%m-%d") if pd.notna(title_end_date) else "N/A"
    ax1.set_title(f'Strategy vs. Nifty (Normalized)\n{title_start_str} to {title_end_str}', fontsize=14)
    ax1.set_ylabel('Normalized Value', fontsize=10); ax1.legend(fontsize=10); ax1.grid(True); ax1.tick_params(axis='x', rotation=30)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1) 
    if 'Close' in nifty_data_df.columns and not nifty_data_df['Close'].empty: 
        nifty_data_df['Close'].plot(ax=ax2, label='Nifty Close Price', alpha=0.8, color='dodgerblue')
    executed_buys = nifty_data_df[nifty_data_df['Trades_Action'] == 1] if 'Trades_Action' in nifty_data_df.columns else pd.DataFrame()
    executed_sells = nifty_data_df[nifty_data_df['Trades_Action'] == -1] if 'Trades_Action' in nifty_data_df.columns else pd.DataFrame()
    if not executed_buys.empty and 'Open' in nifty_data_df.columns : ax2.plot(executed_buys.index, nifty_data_df.loc[executed_buys.index, 'Open'], '^', ms=9, c='forestgreen', label='Buy', alpha=0.9, mec='black')
    if not executed_sells.empty and 'Open' in nifty_data_df.columns: ax2.plot(executed_sells.index, nifty_data_df.loc[executed_sells.index, 'Open'], 'v', ms=9, c='crimson', label='Sell', alpha=0.9, mec='black')
    ax2.set_title('Nifty 50 Price & Trades', fontsize=14); ax2.set_ylabel('Nifty Price', fontsize=10); ax2.legend(fontsize=10); ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1) 
    if 'Sentiment' in nifty_data_df.columns and not nifty_data_df['Sentiment'].empty: 
        nifty_data_df['Sentiment'].plot(ax=ax3, label='Daily Sentiment (T-1 News)', color='lightcoral', lw=1.0, alpha=0.7)
    if 'Rolling_Sentiment' in nifty_data_df.columns and not nifty_data_df['Rolling_Sentiment'].empty:
        if sentiment_lookback_days_param > 1: # Only differentiate if lookback > 1
            nifty_data_df['Rolling_Sentiment'].plot(ax=ax3, label=f'{sentiment_lookback_days_param}-Day Rolling Sentiment', color='mediumpurple', lw=2.0)
        elif 'Sentiment' not in nifty_data_df.columns or nifty_data_df['Sentiment'].empty : # If only rolling is available (lookback=1 makes it same as daily)
             nifty_data_df['Rolling_Sentiment'].plot(ax=ax3, label=f'Sentiment (Signal Source)', color='mediumpurple', lw=2.0)

    ax3.axhline(positive_thresh_val, color='green', linestyle='--', alpha=0.7, label=f'Pos Thresh ({positive_thresh_val})')
    ax3.axhline(negative_thresh_val, color='red', linestyle='--', alpha=0.7, label=f'Neg Thresh ({negative_thresh_val})')
    ax3.axhline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    ax3.set_title('News Sentiment Analysis', fontsize=14); ax3.set_ylabel('VADER Score', fontsize=10); ax3.set_xlabel('Date', fontsize=10)
    ax3.legend(fontsize=10); ax3.grid(True)
    plt.tight_layout(pad=2.5); return fig


# --- STREAMLIT UI ---
st.title(" Nifty 50 News Sentiment Backtester")
if 'system_start_date' not in st.session_state: st.session_state.system_start_date = datetime.now().date()
st.markdown(f"System Date at Session Start (Used for Backtest End Default): **{st.session_state.system_start_date.strftime('%Y-%m-%d')}**")

st.sidebar.header(" API Configuration")
api_key_input = st.sidebar.text_input("NewsAPI Key", value="YOUR_NEWS_API_KEY", type="password", help="Replace with your actual NewsAPI.org key")

st.sidebar.header(" Backtest Period")
default_end_date_val = st.session_state.system_start_date
end_date_ui = st.sidebar.date_input("Backtest End Date", value=default_end_date_val, min_value=date(2020,1,1), max_value=default_end_date_val + timedelta(days=365*2))
days_for_backtest_ui = st.sidebar.number_input("Number of Trading Days for Backtest", min_value=5, max_value=60, value=20, help="Max 60 to manage API calls & runtime.") # Reduced max

st.sidebar.header(" Strategy Parameters")
initial_capital_ui = st.sidebar.number_input("Initial Capital", min_value=10000, value=1000000, step=10000)
st.sidebar.subheader("Sentiment Calculation")
sentiment_lookback_ui = st.sidebar.number_input("Sentiment Lookback (Days)", min_value=1, max_value=7, value=3, help="1 = Daily, >1 = Rolling Avg.") # Reduced max lookback
positive_threshold_ui = st.sidebar.slider("Positive Sentiment Threshold", 0.0, 1.0, 0.15, 0.01) # Adjusted default
negative_threshold_ui = st.sidebar.slider("Negative Sentiment Threshold", -1.0, 1.0, -0.10, 0.01) # Adjusted default
transaction_cost_ui = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01, format="%.2f%%") / 100.0
nifty_ticker_val = "^NSEI"

if st.sidebar.button(" Run Backtest"):
    if not api_key_input or api_key_input == "YOUR_NEWS_API_KEY": st.sidebar.error("Please enter NewsAPI Key.")
    elif not analyzer: st.sidebar.error("Sentiment Analyzer not initialized.")
    else:
        end_date_dt_for_backtest = datetime.combine(end_date_ui, datetime.min.time())
        # Clear caches if parameters change that affect data fetching or client
        # For simplicity, we rely on Streamlit's default caching behavior.
        # For more control, you'd manage st.session_state to clear specific caches.
        
        # Initialize/get the cached NewsAPI client before passing to backtest function
        # This ensures the client is created with the current UI key if it's different.
        # The `run_sentiment_backtest` function will then use this specific client.
        # Note: `get_newsapi_client` is already cached by @st.cache_resource.
        # Passing the key directly to `run_sentiment_backtest` is also fine, as it calls `get_newsapi_client`.

        with st.spinner("Running backtest... This may take a few minutes..."):
            backtest_data_df, performance_results = run_sentiment_backtest(
                api_key_input, nifty_ticker_val, end_date_dt_for_backtest, 
                days_for_backtest_ui, initial_capital_ui, positive_threshold_ui,
                negative_threshold_ui, transaction_cost_ui,
                sentiment_lookback_ui 
            )
        if backtest_data_df is not None and performance_results:
            st.subheader(" Backtest Performance Metrics")
            # ... (Metrics display same as your corrected version) ...
            row1_cols = st.columns(3); row2_cols = st.columns(3)
            row1_cols[0].metric("Final Portfolio Value", f"₹{performance_results.get('final_portfolio_value', 0):,.2f}")
            row1_cols[1].metric("Total Strategy Return", f"{performance_results.get('total_strategy_return_pct', 0):.2f}%")
            bh_return = performance_results.get('nifty_buy_hold_return_pct', 'N/A')
            bh_return_str = f"{bh_return:.2f}%" if isinstance(bh_return, float) else bh_return
            row1_cols[2].metric("Nifty Buy & Hold Return", bh_return_str)
            sharpe = performance_results.get('sharpe_ratio_annualized', 'N/A')
            sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, float) else sharpe
            row2_cols[0].metric("Sharpe Ratio (Annualized)", sharpe_str)
            row2_cols[1].metric("Number of Trades", f"{performance_results.get('num_trades', 0)}")
            row2_cols[2].metric("Initial Capital", f"₹{performance_results.get('initial_capital', 0):,.2f}")


            st.subheader(" Visualizations")
            fig = plot_backtest_results(backtest_data_df, performance_results, initial_capital_ui, positive_threshold_ui, negative_threshold_ui, sentiment_lookback_ui)
            if fig: st.pyplot(fig)

            st.subheader(" Sample Backtest Data (Last 10 Days)")
            if not backtest_data_df.empty:
                display_cols = ['Open', 'Close', 'Sentiment', 'Rolling_Sentiment', 'Signal', 'Position_State', 
                                'Trades_Action', 'Cash', 'Holdings_Units', 
                                'Portfolio_Value', 'Strategy_Return_Daily']
                actual_display_cols = [col for col in display_cols if col in backtest_data_df.columns]
                st.dataframe(backtest_data_df[actual_display_cols].tail(10).round(4))
            else: st.write("No data to display.")
        else: st.error("Backtest failed to produce results.")
else: st.info("Adjust parameters in the sidebar and click 'Run Backtest'.")

st.sidebar.markdown("---")
st.sidebar.info(f"Sentiment Dashboard v0.4 | System Date: {st.session_state.system_start_date.strftime('%Y-%m-%d')}")